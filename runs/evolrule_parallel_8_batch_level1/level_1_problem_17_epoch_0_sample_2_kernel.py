import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_no_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_SIZE 32

__global__ void matmul_no_transpose_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int N, int K) {
    __shared__ float shared_A[TILE_DIM][TILE_DIM + 1];
    __shared__ float shared_B[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    for (int t = 0; t < (K - 1) / TILE_DIM + 1; t++) {
        int a_col = t * TILE_DIM + threadIdx.x;
        int a_row = row;

        int b_row = t * TILE_DIM + threadIdx.y;
        int b_col = col;

        if (a_col < K && a_row < M) {
            shared_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (b_row < K && b_col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[b_col * K + b_row]; // B is stored as row-major (N, K)
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_no_transpose_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K) {
    const int threads = TILE_DIM;
    dim3 block(threads, threads);
    dim3 grid((N + threads - 1) / threads, (M + threads - 1) / threads);

    auto C = torch::empty({M, N}, A.options());

    matmul_no_transpose_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_no_transpose_cpp_source = (
    "torch::Tensor matmul_no_transpose_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);"
)

matmul_no_transpose = load_inline(
    name="matmul_no_transpose",
    cpp_sources=matmul_no_transpose_cpp_source,
    cuda_sources=matmul_no_transpose_source,
    functions=["matmul_no_transpose_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-arch=sm_75"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.M = 1024 * 2
        self.K = 4096 * 2
        self.N = 2048 * 2
        self.matmul_no_transpose = matmul_no_transpose

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_no_transpose.matmul_no_transpose_cuda(A.cuda(), B.cuda(), self.M, self.N, self.K)