import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc) {

    __shared__ float aTile[TILE_SIZE][TILE_SIZE];
    __shared__ float bTile[TILE_SIZE][TILE_SIZE];

    int blockRow = blockIdx.y * TILE_SIZE;
    int blockCol = blockIdx.x * TILE_SIZE;

    int row = blockRow + threadIdx.y;
    int col = blockCol + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load aTile
        int aCol = k * TILE_SIZE + threadIdx.x;
        bool a_valid = (row < M) && (aCol < K);
        aTile[threadIdx.y][threadIdx.x] = a_valid ? A[row * lda + aCol] : 0.0f;

        // Load bTile
        int bRow = k * TILE_SIZE + threadIdx.y;
        int bCol = blockCol + threadIdx.x;
        bool b_valid = (bRow < K) && (bCol < N);
        bTile[threadIdx.y][threadIdx.x] = b_valid ? B[bRow * ldb + bCol] : 0.0f;

        __syncthreads();

        // Compute contribution from this tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
        }

        // Remove the unnecessary __syncthreads()
    }

    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    const int M = A_contig.size(0);
    const int K = A_contig.size(1);
    const int N = B_contig.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::empty({M, N}, options);

    int lda = A_contig.stride(0);
    int ldb = B_contig.stride(0);
    int ldc = C.stride(0);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_kernel<<<grid, block>>>(
        A_contig.data_ptr<float>(),
        B_contig.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc
    );

    cudaDeviceSynchronize();

    return C;
}
"""

matmul_source_cpp = """
#include <torch/extension.h>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_source_cpp,
    cuda_sources=matmul_source_cuda,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda  # The loaded module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)