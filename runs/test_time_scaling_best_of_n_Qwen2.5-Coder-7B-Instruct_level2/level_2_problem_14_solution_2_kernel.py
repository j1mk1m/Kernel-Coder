import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor matrix_multiplication_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    const int grid_x = (n + block_size - 1) / block_size;
    const int grid_y = (m + block_size - 1) / block_size;

    matrix_multiplication_kernel<<<grid_x, grid_y, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

matrix_multiplication_cpp_source = (
    "torch::Tensor matrix_multiplication_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication
matrix_multiplication = load_inline(
    name="matrix_multiplication",
    cpp_sources=matrix_multiplication_cpp_source,
    cuda_sources=matrix_multiplication_source,
    functions=["matrix_multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for division
division_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void division_kernel(const float* a, float* out, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / divisor;
    }
}

torch::Tensor division_cuda(torch::Tensor a, float divisor) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    division_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size, divisor);

    return out;
}
"""

division_cpp_source = (
    "torch::Tensor division_cuda(torch::Tensor a, float divisor);"
)

# Compile the inline CUDA code for division
division = load_inline(
    name="division",
    cpp_sources=division_cpp_source,
    cuda_sources=division_source,
    functions=["division_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for summation
summation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void summation_kernel(const float* a, float* out, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? a[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

torch::Tensor summation_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto out = torch::zeros({1}, a.options());

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    summation_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

summation_cpp_source = (
    "torch::Tensor summation_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for summation
summation = load_inline(
    name="summation",
    cpp_sources=summation_cpp_source,
    cuda_sources=summation_source,
    functions=["summation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for scaling
scaling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaling_kernel(const float* a, float* out, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scale;
    }
}

torch::Tensor scaling_cuda(torch::Tensor a, float scale) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scaling_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size, scale);

    return out;
}
"""

scaling_cpp_source = (
    "torch::Tensor scaling_cuda(torch::Tensor a, float scale);"
)

# Compile the inline CUDA code for scaling
scaling = load_inline(
    name="scaling",
    cpp_sources=scaling_cpp_source,
    cuda_sources=scaling_source,
    functions=["scaling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.matrix_multiplication = matrix_multiplication
        self.division = division
        self.summation = summation
        self.scaling = scaling

    def forward(self, x):
        x = self.matrix_multiplication.matrix_multiplication_cuda(x, self.weight.t())
        x = self.division.division_cuda(x, 2.0)
        x = self.summation.summation_cuda(x)
        x = self.scaling.scaling_cuda(x, self.scaling_factor)
        return x