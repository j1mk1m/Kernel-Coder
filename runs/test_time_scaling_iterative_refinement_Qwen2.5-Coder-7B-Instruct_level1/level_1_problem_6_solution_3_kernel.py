import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    float sum = 0.0f;

    if (row < M && k < K) {
        sA[threadIdx.y][threadIdx.x] = A[row * K + k];
    } else {
        sA[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (col < N && k < K) {
        sB[threadIdx.y][threadIdx.x] = B[k * N + col];
    } else {
        sB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (k < K) {
        for (int i = 0; i < 32; ++i) {
            sum += sA[i][threadIdx.x] * sB[threadIdx.y][i];
        }
    }

    __syncthreads();

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

torch::Tensor matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    const int block_size_x = 32;
    const int block_size_y = 32;
    const int block_size_z = 32;

    const dim3 grid_size((N + block_size_x - 1) / block_size_x, (M + block_size_y - 1) / block_size_y, (K + block_size_z - 1) / block_size_z);
    const dim3 block_size(block_size_x, block_size_y, block_size_z);

    matrix_multiplication_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

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