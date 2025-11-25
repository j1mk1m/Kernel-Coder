import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for min, add bias, and scale operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* input,
    float* output,
    const float constant_value,
    const float* bias,
    const float scaling_factor,
    int N, int C, int H, int W) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) {
        return;
    }

    // Compute the channel index
    int c = (idx / (H * W)) % C;

    float val = input[idx];
    val = min(val, constant_value);
    val += bias[c];
    val *= scaling_factor;

    output[idx] = val;
}

extern "C" {
    torch::Tensor fused_operations_cuda(torch::Tensor input, float constant_value, torch::Tensor bias, float scaling_factor) {
        auto N = input.size(0);
        auto C = input.size(1);
        auto H = input.size(2);
        auto W = input.size(3);

        auto output = torch::empty_like(input);

        const int threads_per_block = 256;
        const int blocks_per_grid = (N * C * H * W + threads_per_block - 1) / threads_per_block;

        fused_operations_kernel<<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            constant_value,
            bias.data_ptr<float>(),
            scaling_factor,
            N, C, H, W
        );

        return output;
    }
}
"""

fused_ops_cpp_source = """
extern "C" {
    torch::Tensor fused_operations_cuda(torch::Tensor input, float constant_value, torch::Tensor bias, float scaling_factor);
}
"""

# Compile the fused CUDA operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops  # Load the fused CUDA operations

    def forward(self, x):
        x = self.conv(x)
        # Apply fused operations in a single kernel
        x = self.fused_ops.fused_operations_cuda(
            x, self.constant_value, self.bias, self.scaling_factor
        )
        return x