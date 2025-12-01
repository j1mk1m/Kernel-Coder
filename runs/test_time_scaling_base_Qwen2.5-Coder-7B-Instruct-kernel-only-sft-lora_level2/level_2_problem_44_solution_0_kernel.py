import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_convolution_source = """
// Your CUDA kernel code here
"""

transposed_convolution_cpp_source = (
    // Your C++ source code here
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution
        self.multiplier = multiplier

    def forward(self, x):
        x = self.transposed_convolution.transposed_convolution_cuda(x, in_channels, out_channels, kernel_size, stride, padding, output_padding)
        x = x * self.multiplier
        x = torch.mean(x, dim=[2, 3], keepdim=True)  # First global average pooling
        x = torch.mean(x, dim=[2, 3], keepdim=True)  # Second global average pooling
        return x