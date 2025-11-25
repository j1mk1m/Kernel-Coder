import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define M 2048
#define K 8192
#define N 4096

#define BLOCK_SIZE 16
#define TILE_SIZE BLOCK_SIZE

__global__ void matrixMulKernel(float* C, const float* A, const float* B) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float Cvalue = 0.0f;

    __shared__ float As[BLOCK_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][BLOCK_SIZE];

    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; m++) {
        // Load A tile
        if (row < M && (m * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + (m * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        if ((m * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply and accumulate
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matrixMulCUDA(torch::Tensor A, torch::Tensor B) {
    const int block_size = BLOCK_SIZE;
    const dim3 threads(block_size, block_size);

    int grid_x = (N + block_size - 1) / block_size;
    int grid_y = (M + block_size - 1) / block_size;
    dim3 grid(grid_x, grid_y);

    auto C = torch::empty({M, N}, A.options());

    matrixMulKernel<<<grid, threads>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>());

    cudaDeviceSynchronize();
    return C;
}
"""

matrix_mul_cpp_source = """
#include <torch/extension.h>

torch::Tensor matrixMulCUDA(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA kernel
matrix_mul = load_inline(
    name="matrix_mul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrixMulCUDA"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixMulCUDA = matrix_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrixMulCUDA.matrixMulCUDA(A, B)

# The given dimensions
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []