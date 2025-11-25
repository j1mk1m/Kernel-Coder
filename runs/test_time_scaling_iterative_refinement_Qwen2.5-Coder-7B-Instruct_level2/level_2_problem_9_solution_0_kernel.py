import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_mul_kernel(const float* a, const float* b, float* c, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; ++k) {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        c[row * cols_b + col] = sum;
    }
}

torch::Tensor matrix_mul_cuda(torch::Tensor a, torch::Tensor b) {
    auto rows_a = a.size(0);
    auto cols_a = a.size(1);
    auto cols_b = b.size(1);
    auto c = torch::zeros({rows_a, cols_b}, a.options());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((cols_b + threads_per_block.x - 1) / threads_per_block.x, (rows_a + threads_per_block.y - 1) / threads_per_block.y);

    matrix_mul_kernel<<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), rows_a, cols_a, cols_b);

    return c;
}
"""

matrix_mul_cpp_source = (
    "torch::Tensor matrix_mul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication
matrix_mul = load_inline(
    name="matrix_mul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for subtraction
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}

torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto c = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);

    return c;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for subtraction
subtraction = load_inline(
    name="subtraction",
    cpp_sources=subtraction_cpp_source,
    cuda_sources=subtraction_source,
    functions=["subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for multiplication
multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void multiplication_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

torch::Tensor multiplication_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto c = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    multiplication_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);

    return c;
}
"""

multiplication_cpp_source = (
    "torch::Tensor multiplication_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for multiplication
multiplication = load_inline(
    name="multiplication",
    cpp_sources=multiplication_cpp_source,
    cuda_sources=multiplication_source,
    functions=["multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* a, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = fmaxf(0.0f, a[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto c = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), c.data_ptr<float>(), size);

    return c;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for ReLU activation
relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.matrix_mul = matrix_mul
        self.subtraction = subtraction
        self.multiplication = multiplication
        self.relu = relu

    def forward(self, x):
        x = self.linear(x)
        x = self.matrix_mul.matrix_mul_cuda(x, x)
        x = self.subtraction.subtraction_cuda(x, torch.full_like(x, self.subtract_value))
        x = self.multiplication.multiplication_cuda(x, torch.full_like(x, self.multiply_value))
        x = self.relu.relu_cuda(x)
        return x


def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features).cuda()]


def get_init_inputs():
    in_features = 8192
    out_features = 8192
    subtract_value = 2.0
    multiply_value = 1.5
    return [in_features, out_features, subtract_value, multiply_value]