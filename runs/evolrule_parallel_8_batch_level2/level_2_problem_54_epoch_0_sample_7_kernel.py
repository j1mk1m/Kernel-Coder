import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void fused_operations_kernel(
    const float* input,
    const float* multiplier,
    float* output,
    int batch,
    int channels,
    int height,
    int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * height * width)
        return;

    // Compute the channel index
    int c = (idx / (height * width)) % channels;

    // Get the multiplier value for this channel
    float m = multiplier[c]; // multiplier is [channels][1][1]

    // Get input value at idx
    float x = input[idx];

    // Multiply by multiplier
    x *= m;

    // Apply Leaky ReLU
    x = fmaxf(x, 0.01f * x);

    // Apply GELU approximation
    const float a = 0.7978845608f;
    const float b = 0.044715f;
    float tmp = x + b * x * x * x;
    tmp *= a;
    float tanh_val = tanhf(tmp);
    float gelu_val = 0.5f * x * (1.0f + tanh_val);

    output[idx] = gelu_val;
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor multiplier) {
    // Check dimensions
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    // The multiplier should have shape (channels, 1, 1)
    TORCH_CHECK(multiplier.size(0) == channels);
    TORCH_CHECK(multiplier.size(1) == 1 && multiplier.size(2) == 1);

    // Ensure multiplier is contiguous
    multiplier = multiplier.contiguous();

    // Output tensor has same shape as input
    auto output = torch::empty_like(input);

    int size = input.numel();

    const int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    fused_operations_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, height, width
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor multiplier);
"""

# Compile the fused kernel
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_operations = fused_operations  # Loaded CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_operations.fused_operations_cuda(x, self.multiplier)
        return x

# The original get_inputs and get_init_inputs remain unchanged
batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]