import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for your operation
custom_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Your CUDA kernel implementation here
"""

custom_op_cpp_source = (
    // Function signature for your custom operation
)

# Compile the inline CUDA code for your operation
custom_op = load_inline(
    name="custom_op_name",
    cpp_sources=custom_op_cpp_source,
    cuda_sources=custom_op_source,
    functions=["custom_op_function_name"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.custom_op = custom_op

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract_value
        x = self.custom_op.custom_op_function_name(x)
        x = self.pool(x)
        x = self.custom_op.custom_op_function_name(x)
        return x