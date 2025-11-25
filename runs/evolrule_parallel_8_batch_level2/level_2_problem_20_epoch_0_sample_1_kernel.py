import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* y0_data,
    const float* bias_data,
    float* out_data,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * depth * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % channels;
    int n = idx / (channels * depth * height * width);

    float y0_val = y0_data[idx];
    float bias_val = bias_data[c]; // bias is [C,1,1,1], so access via c

    float temp = (y0_val + bias_val) + y0_val;
    temp *= y0_val;
    temp += y0_val;
    out_data[idx] = temp;
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor y0,
    torch::Tensor bias) {

    const int batch_size = y0.size(0);
    const int channels = y0.size(1);
    const int depth = y0.size(2);
    const int height = y0.size(3);
    const int width = y0.size(4);

    auto out = torch::empty_like(y0);

    const int num_elements = batch_size * channels * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        y0.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, channels, depth, height, width);

    return out;
}
"""

elementwise_fused_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor y0, torch::Tensor bias);"
)

elementwise_fused = load_inline(
    name="elementwise_fused",
    cpp_sources=elementwise_fused_cpp_source,
    cuda_sources=elementwise_fused_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_elementwise = elementwise_fused  # Load the fused kernel

    def forward(self, x):
        y0 = self.conv_transpose(x)
        return self.fused_elementwise.fused_elementwise_cuda(y0, self.bias)