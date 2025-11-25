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

    const int block_size = 32;
    dim3 grid((cols_b + block_size - 1) / block_size, (rows_a + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    matrix_mul_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), rows_a, cols_a, cols_b);

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


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.matrix_mul = matrix_mul

    def forward(self, x):
        x = self.linear(x)
        x = x / self.divisor
        x = self.matrix_mul.matrix_mul_cuda(x, x)
        x = torch.nn.functional.gelu(x)
        return x