import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Your CUDA kernels go here
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the 3D convolution kernel here
"""

leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the LeakyReLU kernel here
"""

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the GELU kernel here
"""

sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the Sigmoid kernel here
"""

add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the add_bias kernel here
"""

# Compile the inline CUDA code
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_source,
    cuda_sources=convolution_source,
    functions=["convolution_3d"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=leaky_relu_source,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_3d"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_source,
    cuda_sources=gelu_source,
    functions=["gelu_3d"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

sigmoid = load_inline(
    name="sigmoid",
    cpp_sources=sigmoid_source,
    cuda_sources=sigmoid_source,
    functions=["sigmoid_3d"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

add_bias = load_inline(
    name="add_bias",
    cpp_sources=add_bias_source,
    cuda_sources=add_bias_source,
    functions=["add_bias_3d"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv.convolution_3d(x, self.conv.weight, self.conv.bias)
        x = leaky_relu.leaky_relu_3d(x, negative_slope=0.01)
        x = gelu.gelu_3d(x)
        x = sigmoid.sigmoid_3d(x)
        x = x + self.bias
        return x