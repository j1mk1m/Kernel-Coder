import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for padding
padding_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void padding_kernel(float* data, int batch_size, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len) {
        return;
    }

    int batch_id = idx / seq_len;
    int seq_idx = idx % seq_len;

    if (seq_idx == 0) {
        data[idx] = 0.0f; // Add zero at the beginning of each sequence
    } else {
        data[idx] = data[idx - 1]; // Copy previous value
    }
}
"""

padding_cpp_source = (
    "void padding_cuda(torch::Tensor data, int batch_size, int seq_len, int dim);"
)

# Compile the inline CUDA code for padding
padding = load_inline(
    name="padding",
    cpp_sources=padding_cpp_source,
    cuda_sources=padding_source,
    functions=["padding_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void exclusive_cumsum_kernel(float* data, int batch_size, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len) {
        return;
    }

    int batch_id = idx / seq_len;
    int seq_idx = idx % seq_len;

    if (seq_idx > 0) {
        data[idx] += data[idx - 1]; // Compute exclusive cumulative sum
    }
}
"""

exclusive_cumsum_cpp_source = (
    "void exclusive_cumsum_cuda(torch::Tensor data, int batch_size, int seq_len, int dim);"
)

# Compile the inline CUDA code for exclusive cumulative sum
exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Step 1: Padding
        padding_cuda(x, batch_size, seq_len, self.dim)

        # Step 2: Exclusive Cumulative Sum
        exclusive_cumsum_cuda(x, batch_size, seq_len, self.dim)

        return x[:, :-1]  # Remove the last element added during padding