import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel (add, multiply, sigmoid)
elementwise_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_elementwise_kernel(
    const float* input, const float* bias, const float* scale,
    float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    float b = bias[c];
    float s = scale[c];
    float val = input[idx] + b;
    val *= s;
    output[idx] = 1.0 / (1.0 + expf(-val));  // Sigmoid activation
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    assert(bias.size(0) == channels && bias.size(1) == 1 && bias.size(2) == 1);
    assert(scale.size(0) == channels && scale.size(1) == 1 && scale.size(2) == 1);

    auto output = torch::empty_like(input);

    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width);

    return output;
}
"""

elementwise_fused_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);"
)

# Compile the fused element-wise kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=elementwise_fused_cpp_source,
    cuda_sources=elementwise_fused_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_elementwise = fused_elementwise  # Load the compiled kernel

    def forward(self, x):
        x = self.conv(x)  # Apply convolution
        # Apply fused element-wise operations (add bias, scale, sigmoid)
        x = self.fused_elementwise.fused_elementwise_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)  # Apply group normalization
        return x