import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matmul_cuda_kernel(float* C, float* A, float* B, int M, int K, int N) {
    __shared__ float ds_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float ds_B[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float Cvalue = 0.0;

    for (int block = 0; block < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++block) {
        int a_col = block * BLOCK_SIZE + tx;
        int a_row = row;
        if (a_row < M && a_col < K) {
            ds_A[ty][tx] = A[a_row * K + a_col];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        int b_row = block * BLOCK_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            ds_B[ty][tx] = B[b_row * N + b_col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_x, grid_y);

    matmul_cuda_kernel<<<grid, threads>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)