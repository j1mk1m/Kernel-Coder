import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and bias addition CUDA kernel
fused_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_relu_bias_kernel(
    const float* input, const float* bias, float* output,
    int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (width * height * channels);

    float val = input[idx];
    val = fmaxf(val, 0.0f);
    val += bias[c];  // Bias is (channels, 1, 1), so access per channel
    output[idx] = val;
}

torch::Tensor fused_relu_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // Validate bias dimensions
    if (bias.sizes()[0] != channels || bias.sizes()[1] != 1 || bias.sizes()[2] != 1) {
        AT_ERROR("Bias tensor must have shape (", channels, ", 1, 1)");
    }

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    int total_elements = batch_size * channels * height * width;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_relu_bias_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width);

    return output;
}
"""

fused_relu_bias_cpp_source = (
    "torch::Tensor fused_relu_bias_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the fused ReLU and bias addition kernel
fused_relu_bias = load_inline(
    name="fused_relu_bias",
    cuda_sources=fused_relu_bias_source,
    cpp_sources=fused_relu_bias_cpp_source,
    functions=["fused_relu_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_relu_bias = fused_relu_bias

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_bias.fused_relu_bias_cuda(x, self.bias)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

# Constants from the original problem setup
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)