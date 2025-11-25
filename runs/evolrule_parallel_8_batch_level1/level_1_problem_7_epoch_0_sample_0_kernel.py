import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMulKernel(float* C, const float* A, const float* B, int M, int K, int N) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Cvalue = 0.0f;

    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    for (int k = 0; k < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++k) {
        int aCol = k * TILE_WIDTH + threadIdx.x;
        int bRow = k * TILE_WIDTH + threadIdx.y;

        // Load tiles into shared memory
        if (row < M && aCol < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum over the tile
        for (int n = 0; n < TILE_WIDTH; ++n) {
            Cvalue += sA[threadIdx.y][n] * sB[n][threadIdx.x];
        }

        __syncthreads(); // Wait until all threads finish processing the tile before next iteration
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    assert(K == B.size(0));

    auto C = torch::empty({M, N}, torch::device("cuda").dtype(torch::kFloat32));

    int block_size = TILE_WIDTH;
    dim3 threads(block_size, block_size);
    dim3 blocks(
        (N + block_size - 1) / block_size,
        (M + block_size - 1) / block_size
    );

    matrixMulKernel<<<blocks, threads>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), M, K, N);

    return C;
}
"""

matrix_mul_cpp_source = """
torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B);
"""

matrix_mul = load_inline(
    name="matrix_mul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mul = matrix_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda()
        B = B.cuda()
        return self.matrix_mul.matrix_mul_cuda(A, B)