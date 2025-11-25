import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* x_data,
    const float* bias_data,
    float scaling_factor,
    float* out_data,
    int size,
    int channels,
    int height,
    int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Calculate indices
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (channels * width * height);

    // Compute the value step by step
    float value = x_data[idx] + bias_data[c]; // Add bias per channel
    value = fmaxf(0.0f, fminf(value, 1.0f)); // First clamp [0,1]
    value *= scaling_factor;                 // Scale
    value = fmaxf(0.0f, fminf(value, 1.0f)); // Second clamp [0,1]
    value /= scaling_factor;                 // Unscale

    out_data[idx] = value;
}

torch::Tensor fused_operations_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor) {
    auto channels = bias.size(0);
    auto height = x.size(2);
    auto width = x.size(3);
    auto size = x.numel();

    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    // Launch the kernel
    fused_operations_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        out.data_ptr<float>(),
        size,
        channels,
        height,
        width
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_operations_cuda(torch::Tensor x, torch::Tensor bias, float scaling_factor);"
)

# Compile the fused operations CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops  # The CUDA module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_operations_cuda(x, self.bias, self.scaling_factor)
        return x

# The get_init_inputs and get_inputs functions remain the same as original
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]