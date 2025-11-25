import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, const float* bias, float* output, int in_channels, int out_channels, int height, int width, int kernel_size, int padding, int stride, int dilation) {
    // Your convolution kernel implementation here
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding, int stride, int dilation) {
    auto batch_size = input.size(0);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), in_channels, out_channels, in_height, in_width, kernel_size, padding, stride, dilation);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding, int stride, int dilation);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.divisor = divisor

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, self.bias, padding=1, stride=1, dilation=1)
        x = x / self.divisor
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        return x