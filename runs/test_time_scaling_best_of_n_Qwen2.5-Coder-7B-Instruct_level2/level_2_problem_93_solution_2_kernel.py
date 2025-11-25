import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom transposed convolution kernel implementation here

torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, int stride, int padding, int output_padding, int groups) {
    // Kernel implementation details here
}
"""

transposed_convolution_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, int stride, int padding, int output_padding, int groups);"
)

# Compile the inline CUDA code for transposed convolution
transposed_convolution = load_inline(
    name="transposed_convolution",
    cpp_sources=transposed_convolution_cpp_source,
    cuda_sources=transposed_convolution_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.transposed_convolution.transposed_convolution_cuda(x, self.weight, self.bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups)
        x = x + self.add_value
        x = torch.min(x, torch.tensor(0.0, device=x.device))
        x = torch.nn.functional.gelu(x)
        x = x * self.multiply_value
        return x