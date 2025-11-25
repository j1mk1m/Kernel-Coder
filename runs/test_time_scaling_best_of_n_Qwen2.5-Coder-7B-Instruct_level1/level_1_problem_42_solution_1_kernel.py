import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max pooling 2D
max_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the custom CUDA kernel for max pooling 2D here
// Your implementation should include the logic for iterating over the input tensor,
// applying the max pooling operation, and storing the results in the output tensor.

// Example pseudo-code:
// for each output pixel position (o_h, o_w):
//     max_value = -INFINITY
//     for each input pixel position within the pooling window (i_h, i_w):
//         value = input[i_h, i_w]
//         if value > max_value:
//             max_value = value
//     output[o_h, o_w] = max_value

// Note: You will need to handle the edge cases, such as when the input tensor dimensions
// are not divisible by the pooling window dimensions without remainder.

// Hint: Use shared memory to improve performance for large pooling windows.

// Pseudo-code end
"""

max_pool_2d_cpp_source = (
    "torch::Tensor max_pool_2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for max pooling 2D
max_pool_2d = load_inline(
    name="max_pool_2d",
    cpp_sources=max_pool_2d_cpp_source,
    cuda_sources=max_pool_2d_source,
    functions=["max_pool_2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_pool_2d.max_pool_2d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)