import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels here...

# Example custom CUDA kernel for transposed 3D convolution
transposed_convolution_source = """
// Your CUDA kernel code for transposed 3D convolution goes here...
"""

# Compile the custom CUDA kernel for transposed 3D convolution
transposed_convolution = load_inline(
    name="transposed_convolution",
    cpp_sources="",
    cuda_sources=transposed_convolution_source,
    functions=["your_transposed_convolution_function"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define custom CUDA kernels for other operations similarly...

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution
        # Initialize other custom CUDA kernels...

    def forward(self, x):
        x = self.transposed_convolution.your_transposed_convolution_function(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x

# Initialize the model with the same parameters as before...