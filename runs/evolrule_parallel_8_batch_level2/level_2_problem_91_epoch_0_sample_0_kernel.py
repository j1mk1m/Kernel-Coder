import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_element_wise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_element_wise_kernel(
    const float* input, const float* bias, float scaling_factor,
    float* output,
    int batch_size, int channels, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    float bias_val = bias[c]; // Access the bias for current channel

    float value = input[idx];
    value = (value + bias_val) * scaling_factor;
    output[idx] = 1.0f / (1.0f + expf(-value));
}

torch::Tensor fused_element_wise_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor) {

    input = input.contiguous();
    auto output = torch::empty_like(input);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int numel = batch_size * channels * height * width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (numel + threads_per_block - 1) / threads_per_block;

    fused_element_wise_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );

    return output;
}
"""

fused_element_wise_cpp_source = (
    "torch::Tensor fused_element_wise_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"
)

# Compile the fused element-wise CUDA kernel
fused_element_wise = load_inline(
    name="fused_element_wise",
    cpp_sources=fused_element_wise_cpp_source,
    cuda_sources=fused_element_wise_source,
    functions=["fused_element_wise_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_element_wise = fused_element_wise  # Reference to the loaded kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.softmax(x, dim=1)
        x = self.fused_element_wise.fused_element_wise_cuda(x, self.bias, self.scaling_factor)
        return x

# The get_inputs and get_init_inputs functions are unchanged from the original.
# They are included here for completeness but not part of the ModelNew code.
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        scaling_factor,
    ]