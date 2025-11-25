import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_operations_kernel(
    const float* input,
    float scaling_factor,
    float min_val,
    float max_val,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] * scaling_factor;
        temp = fmaxf(fminf(temp, max_val), min_val);
        float x = temp;
        float poly = 0.044715f * x * x * x;
        float tanh_term = tanhf( sqrt(2.0f / M_PI) * (x + poly) );
        output[idx] = 0.5f * x * (1.0f + tanh_term);
    }
}

torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    float scaling_factor,
    float min_val,
    float max_val
) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scaling_factor,
        min_val,
        max_val,
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

fused_ops_cpp = """
torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    float scaling_factor,
    float min_val,
    float max_val
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.gemm(x)
        return self.fused_ops.fused_operations_cuda(
            x, self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]