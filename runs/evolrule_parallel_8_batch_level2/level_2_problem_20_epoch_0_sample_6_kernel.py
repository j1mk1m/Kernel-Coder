import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_elementwise_source = """
#include <torch/extension.h>

__global__ void fused_elementwise_kernel(
    const float* y_data,
    const float* bias_data,
    float* out_data,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int D_H_W = depth * height * width;
    int temp = idx / D_H_W;
    int c = temp % out_channels;

    float y = y_data[idx];
    float bias = bias_data[c]; // Access the bias for the current channel

    out_data[idx] = y * (2 * y + bias + 1);
}

torch::Tensor fused_elementwise_cuda(torch::Tensor y, torch::Tensor bias) {
    auto options = torch::TensorOptions().dtype(y.dtype()).device(y.device());
    auto out = torch::empty_like(y);

    int batch_size = y.size(0);
    int out_channels = y.size(1);
    int depth = y.size(2);
    int height = y.size(3);
    int width = y.size(4);

    int num_elements = batch_size * out_channels * depth * height * width;

    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_elementwise_kernel<<<blocks_per_grid, threads_per_block>>>(
        y.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_channels,
        depth,
        height,
        width
    );

    return out;
}
"""

fused_elementwise_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor y, torch::Tensor bias);"
)

# Compile the CUDA extension
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=[fused_elementwise_cpp_source],
    cuda_sources=[fused_elementwise_source],
    functions=["fused_elementwise_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Load the fused element-wise function
        self.fused_elementwise = fused_elementwise

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_elementwise.fused_elementwise_cuda(x, self.bias)
        return x