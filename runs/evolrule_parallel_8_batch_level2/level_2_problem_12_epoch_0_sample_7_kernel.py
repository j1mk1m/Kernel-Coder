import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Multiply and LeakyReLU operation
fused_mul_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mul_leakyrelu_kernel(
    const float* in,
    float* out,
    float multiplier,
    float negative_slope,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in[idx] * multiplier;
        out[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

torch::Tensor fused_mul_leakyrelu_cuda(
    torch::Tensor in,
    float multiplier,
    float negative_slope) {
    int size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mul_leakyrelu_kernel<<<num_blocks, block_size>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        multiplier,
        negative_slope,
        size);

    return out;
}
"""

fused_mul_leakyrelu_cpp_source = (
    "torch::Tensor fused_mul_leakyrelu_cuda(torch::Tensor in, float multiplier, float negative_slope);"
)

# Load the fused CUDA kernel
fused_mul_leakyrelu = load_inline(
    name="fused_mul_leakyrelu",
    cpp_sources=fused_mul_leakyrelu_cpp_source,
    cuda_sources=fused_mul_leakyrelu_source,
    functions=["fused_mul_leakyrelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused = fused_mul_leakyrelu  # Load the fused kernel

    def forward(self, x):
        x = self.gemm(x)
        # Apply fused multiply and LeakyReLU
        return self.fused.fused_mul_leakyrelu_cuda(
            x, self.multiplier, self.negative_slope
        )

# Global parameters (same as original)
batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]