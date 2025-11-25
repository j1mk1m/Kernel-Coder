import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with shared memory
matrix_multiplication_shared_memory_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

__global__ void matrix_multiplication_shared_memory_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // Load A and B into shared memory
        if (row < N && m * BLOCK_SIZE + threadIdx.x < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + m * BLOCK_SIZE + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && m * BLOCK_SIZE + threadIdx.y < N) {
            s_B[threadIdx.x][threadIdx.y] = B[(m * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Perform partial reduction within shared memory
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_multiplication_shared_memory_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_multiplication_shared_memory_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matrix_multiplication_shared_memory_cpp_source = (
    "torch::Tensor matrix_multiplication_shared_memory_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication with shared memory
matrix_multiplication_shared_memory = load_inline(
    name="matrix_multiplication_shared_memory",
    cpp_sources=matrix_multiplication_shared_memory_cpp_source,
    cuda_sources=matrix_multiplication_shared_memory_source,
    functions=["matrix_multiplication_shared_memory_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_multiplication_shared_memory = matrix_multiplication_shared_memory

    def forward(self, A, B):
        return self.matrix_multiplication_shared_memory.matrix_multiplication_shared_memory_cuda(A, B)