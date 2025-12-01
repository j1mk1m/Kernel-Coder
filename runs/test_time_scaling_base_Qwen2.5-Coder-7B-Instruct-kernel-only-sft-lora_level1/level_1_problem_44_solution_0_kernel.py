import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
// Your CUDA kernel code here
"""

avg_pool_1d_cpp_source = (
    // Your C++ source code here
)

# Compile the inline CUDA code for 1D Average Pooling
avg_pool_1d = load_inline(
    name="avg_pool_1d",
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_source,
    functions=["avg_pool_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = avg_pool_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool.avg_pool_1d_cuda(x, self.kernel_size, self.stride, self.padding)