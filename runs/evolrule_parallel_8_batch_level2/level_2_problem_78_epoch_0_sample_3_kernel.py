import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1_kernel_size = 2
        self.max_pool1_stride = 2
        self.max_pool2_kernel_size = 3
        self.max_pool2_stride = 3

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool3d_cuda(x, self.max_pool1_kernel_size, self.max_pool1_stride)
        x = self.max_pool3d_cuda(x, self.max_pool2_kernel_size, self.max_pool2_stride)
        x = self.sum_dim1_cuda(x)
        return x

# Define the CUDA sources
max_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// ... (the kernel code as before)
"""

max_pool3d_cpp = """
torch::Tensor max_pool3d_cuda(torch::Tensor input, int kernel_size, int stride);
torch::Tensor sum_dim1_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
max_pool3d_module = load_inline(
    name="max_pool3d",
    cpp_sources=max_pool3d_cpp,
    cuda_sources=max_pool3d_source,
    functions=["max_pool3d_cuda", "sum_dim1_cuda"],
    verbose=True,
)

# Assign the functions to the ModelNew class
ModelNew.max_pool3d_cuda = max_pool3d_module.max_pool3d_cuda
ModelNew.sum_dim1_cuda = max_pool3d_module.sum_dim1_cuda