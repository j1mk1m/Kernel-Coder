import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu(float x) {
    const float a = 0.7978845608f;
    const float b = 0.044715f;
    float x_cubed = x * x * x;
    float z = a * (x + b * x_cubed);
    return 0.5f * x * (1.0f + tanhf(z));
}

__global__ void fused_operations_kernel(
    const float* input,
    float* output,
    int size,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] * scaling_factor;
        temp = fmaxf(hardtanh_min, fminf(hardtanh_max, temp));
        output[idx] = gelu(temp);
    }
}

torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    float scaling_factor,
    float hardtanh_min,
    float hardtanh_max) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        scaling_factor,
        hardtanh_min,
        hardtanh_max);

    return output;
}
"""

elementwise_fused_cpp_source = (
    "torch::Tensor fused_operations_cuda(torch::Tensor input, float scaling_factor, float hardtanh_min, float hardtanh_max);"
)

# Compile the inline CUDA code
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=elementwise_fused_cpp_source,
    cuda_sources=elementwise_fused_source,
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
        self.fused_ops = fused_operations

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_ops.fused_operations_cuda(
            x,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )
        return x

# Global parameters as in the original model
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