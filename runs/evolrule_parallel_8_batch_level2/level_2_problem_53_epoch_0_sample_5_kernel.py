import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_kernel(const float* x, float* out,
    float scaling_factor, float hardtanh_min, float hardtanh_max,
    int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float temp = x[idx] * scaling_factor;
    temp = fmaxf(hardtanh_min, fminf(hardtanh_max, temp));

    float inner = sqrt(2.0f / M_PI) * (temp + 0.044715f * temp * temp * temp);
    float tanh_val = tanhf(inner);
    float gelu_val = 0.5f * temp * (1.0f + tanh_val);

    out[idx] = gelu_val;
}

torch::Tensor fused_function(torch::Tensor x,
    float scaling_factor, float hardtanh_min, float hardtanh_max) {

    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        scaling_factor, hardtanh_min, hardtanh_max, size
    );

    return out;
}
"""

fused_cpp_header = (
    "torch::Tensor fused_function(torch::Tensor x, float scaling_factor, float hardtanh_min, float hardtanh_max);"
)

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_ops",
    cpp_sources=fused_cpp_header,
    cuda_sources=fused_kernel_source,
    functions=["fused_function"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.gemm(x)
        return fused_op.fused_function(
            x, self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )

# Ensure the same initialization as before
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