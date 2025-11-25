import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* conv_out, const float* bias, float* out,
    int batch_size, int out_channels, int depth, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % out_channels;
    int n = idx / (out_channels * depth * height * width);

    float b = bias[c];
    float C = conv_out[idx];
    float term = 2 * C + b + 1;
    out[idx] = C * term;
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor conv_out, torch::Tensor bias) {
    int batch_size = conv_out.size(0);
    int out_channels = conv_out.size(1);
    int depth = conv_out.size(2);
    int height = conv_out.size(3);
    int width = conv_out.size(4);

    auto out = torch::empty_like(conv_out);

    int num_elements = batch_size * out_channels * depth * height * width;

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, out_channels, depth, height, width);

    return out;
}
"""

fused_elementwise_cpp_source = "torch::Tensor fused_elementwise_cuda(torch::Tensor conv_out, torch::Tensor bias);"

# Compile the fused element-wise kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp_source,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_elementwise = fused_elementwise

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_elementwise.fused_elementwise_cuda(x, self.bias)