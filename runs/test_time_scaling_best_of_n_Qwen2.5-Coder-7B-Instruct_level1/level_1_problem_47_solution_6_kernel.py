import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(const float* input, float* output, int n_elements) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < n_elements) {
        shared[tid] = input[i];
    } else {
        shared[tid] = 0.0f;
    }

    __syncthreads();

    // Perform reduction within shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Write result for current block to global memory
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

void sum_reduction_cuda(const float* input, float* output, int n_elements, int block_size) {
    int grid_size = (n_elements + block_size - 1) / block_size;

    sum_reduction_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(input, output, n_elements);
}
"""

sum_reduction_cpp_source = (
    "void sum_reduction_cuda(const float* input, float* output, int n_elements, int block_size);"
)

# Compile the inline CUDA code for sum reduction
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_elements = x.size(self.dim)
        output = torch.zeros(1, dtype=torch.float32, device=x.device)

        sum_reduction.sum_reduction_cuda(x.data_ptr(), output.data_ptr(), n_elements, 256)

        return output.view(1, 1, *x.shape[self.dim+1:])