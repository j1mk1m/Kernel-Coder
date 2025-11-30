import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max reduction
max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_reduction_kernel(const float* data, float* result, int numel) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < numel) ? data[i] : -std::numeric_limits<float>::infinity();
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(result + blockIdx.x, sdata[0]);
    }
}

void max_reduction_cuda(const float* data, float* result, int numel, int blocks) {
    const int threads_per_block = 256;

    max_reduction_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(data, result, numel);
}
"""

max_reduction_cpp_source = (
    "void max_reduction_cuda(const float* data, float* result, int numel, int blocks);"
)

# Compile the inline CUDA code for max reduction
max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        reduced_shape = list(x.shape)
        reduced_shape[self.dim] = 1
        result = torch.zeros(reduced_shape, device=x.device, dtype=x.dtype)

        batch_size = x.size(0)
        max_reduction_cuda(x.contiguous().view(-1).data_ptr(), result.view(-1).data_ptr(), batch_size, batch_size)

        return result.squeeze(self.dim)