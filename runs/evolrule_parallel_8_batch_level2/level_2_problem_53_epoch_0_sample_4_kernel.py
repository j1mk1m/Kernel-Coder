import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA fused activation kernel code
fused_activation_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

#define M_SQRT2_OVER_PI 0.7978845608f

__global__ void fused_activation_kernel(
    const float* input,
    float* output,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * scaling_factor;
        val = val < hardtanh_min ? hardtanh_min : (val > hardtanh_max ? hardtanh_max : val);
        float x = val;
        float inner = M_SQRT2_OVER_PI * (x + 0.044715f * x * x * x);
        val = 0.5f * x * (1.0f + tanhf(inner));
        output[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input, float scaling_factor, float hardtanh_min, float hardtanh_max) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        hardtanh_min,
        hardtanh_max,
        size
    );

    return output;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor input, float scaling_factor, float hardtanh_min, float hardtanh_max);"
)

# Load the fused activation kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_activation = fused_activation  # The loaded CUDA module

    def forward(self, x):
        x = self.gemm(x)
        return self.fused_activation.fused_activation_cuda(
            x, self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]