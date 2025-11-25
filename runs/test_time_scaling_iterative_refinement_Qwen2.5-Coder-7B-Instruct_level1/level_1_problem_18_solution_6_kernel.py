import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 16;
    dim3 grid_dim((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
    dim3 block_dim(block_size, block_size);

    matrix_multiplication_kernel<<<grid_dim, block_dim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matrix_multiplication_cpp_source = (
    "torch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B);"
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


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_multiplication = matrix_multiplication

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_multiplication.matrix_multiplication_cuda(A, B)