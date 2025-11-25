import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        if (row < M && k * BLOCK_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && k * BLOCK_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int m = 0; m < BLOCK_SIZE; ++m) {
            sum += As[threadIdx.y][m] * Bs[m][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    matrix_multiplication_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

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
        """
        Performs matrix multiplication using custom CUDA kernel.
        """
        return self.matrix_multiplication.matrix_multiplication_cuda(A, B)