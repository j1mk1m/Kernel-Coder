import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Global variables defining the model parameters
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

# Define the fused ReLU + bias addition CUDA kernel
fused_relu_add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_add_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int out_channels,
    int height,
    int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;

    // Compute indices
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int n = idx / (out_channels * width * height);

    float val = input[idx];
    val = fmaxf(val, 0.f); // ReLU
    val += bias[c]; // bias is [out_channels, 1, 1], so c is the channel index
    output[idx] = val;
}

torch::Tensor fused_relu_add_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int out_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = torch::empty_like(input);

    const int total_elements = batch_size * out_channels * height * width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_relu_add_bias_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        height,
        width);

    return output;
}
"""

# The corresponding C++ declaration
fused_relu_add_cpp_source = """
extern "C" {
    torch::Tensor fused_relu_add_bias_cuda(torch::Tensor input, torch::Tensor bias);
}
"""

# Compile the CUDA extension
fused_relu_add = load_inline(
    name="fused_relu_add",
    cpp_sources=[fused_relu_add_cpp_source],
    cuda_sources=[fused_relu_add_bias_source],
    functions=["fused_relu_add_bias_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_relu_add = fused_relu_add  # Access the module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_add.fused_relu_add_bias_cuda(x, self.bias)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]