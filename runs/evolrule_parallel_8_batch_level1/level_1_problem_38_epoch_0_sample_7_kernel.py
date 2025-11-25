import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l1_norm_kernel(float* x, float* out, int batch_size, int dim) {
    int row = blockIdx.x;
    __shared__ float sdata[256]; // Shared memory for partial sums

    float sum_partial = 0.0f;

    // Compute partial sum of absolute values for the row
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        float val = x[idx];
        sum_partial += fabsf(val);
    }

    // Write partial sum to shared memory
    sdata[threadIdx.x] = sum_partial;
    __syncthreads();

    // Block reduction to compute sum_row
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum_row = sdata[0];
    __syncthreads();

    float denominator = dim / sum_row;

    // Compute normalized values
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        float val = x[idx];
        out[idx] = val * denominator;
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    x = x.contiguous();
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::empty_like(x);

    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;

    l1_norm_kernel<<<blocks_per_grid, threads_per_block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return out;
}
"""

l1_norm_cpp_source = """
#include <torch/extension.h>

torch::Tensor l1_norm_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name="l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["l1_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_norm = l1_norm  # Reference to the CUDA module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)