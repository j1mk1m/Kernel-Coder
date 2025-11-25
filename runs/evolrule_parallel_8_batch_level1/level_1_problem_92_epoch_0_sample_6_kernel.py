import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for shifting the cumulative sum to get exclusive values
shift_left_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void shift_left_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size,
    int dim_size,
    int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim_size) return;

    int batch = idx / dim_size;
    int pos = idx % dim_size;

    if (dim == 1) {
        if (pos == 0) {
            output[idx] = 0;
        } else {
            output[idx] = input[batch * dim_size + (pos - 1)];
        }
    }
}

void shift_left_cuda(torch::Tensor input, torch::Tensor output, int dim) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1);
    const int num_elements = batch_size * dim_size;

    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "shift_left_cuda", ([&] {
        shift_left_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            dim
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
}
"""

shift_left_cpp_source = "void shift_left_cuda(torch::Tensor input, torch::Tensor output, int dim);"

# Compile the CUDA kernel
shift_left = load_inline(
    name="shift_left",
    cpp_sources=shift_left_cpp_source,
    cuda_sources=shift_left_source,
    functions=["shift_left_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Compute inclusive cumulative sum using PyTorch's optimized cumsum
        inclusive = torch.cumsum(x, dim=self.dim)
        # Create an output tensor with the same shape as inclusive
        output = torch.empty_like(inclusive)
        # Apply the shift kernel to get the exclusive cumulative sum
        shift_left.shift_left_cuda(inclusive, output, self.dim)
        return output