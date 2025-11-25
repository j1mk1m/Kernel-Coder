import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

import math

# Custom CUDA kernel for GELU approximation using the exact formulation
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Use fast approximation for tanh with __tanhf_approx if available
#ifndef __TANHF_APPROX
#define __TANHF_APPROX
__device__ float fast_tanh(float x) {
    return tanh(x);
}
#else
__device__ float fast_tanh(float x) {
    return __tanhf_approx(x);
}
#endif

__global__ void gelu_kernel(const float* x, float* y, int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float xi = x[tid];
        float x_cubed = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * x_cubed);
        float tanh_val = fast_tanh(inner);
        y[tid] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    int elements = x.numel();
    int threads_per_block = 256;
    int blocks_per_grid = (elements + threads_per_block - 1) / threads_per_block;

    auto y = torch::empty_like(x);

    gelu_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), elements);
    return y;
}
"""

gelu_cuda_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the CUDA kernel
gelu_cuda = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cuda_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=["-D__TANHF_APPROX"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_cuda = gelu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_cuda.gelu_cuda(x)