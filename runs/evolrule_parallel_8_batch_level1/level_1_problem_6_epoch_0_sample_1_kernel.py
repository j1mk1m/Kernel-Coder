import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrix_mult_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockRow = blockIdx.y * TILE_WIDTH;
    int blockCol = blockIdx.x * TILE_WIDTH;

    int row = blockRow + ty;
    int col = blockCol + tx;

    float Cvalue = 0.0f;

    for (int i = 0; i < (K + TILE_WIDTH - 1)/TILE_WIDTH; i++) {
        // Load A into shared memory
        int aRow = blockRow + ty;
        int aCol = i * TILE_WIDTH + tx;
        if (aRow < M && aCol < K) {
            sA[ty][tx] = A[aRow * K + aCol];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B into shared memory
        int bRow = i * TILE_WIDTH + ty;
        int bCol = blockCol + tx;
        if (bRow < K && bCol < N) {
            sB[ty][tx] = B[bRow * N + bCol];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the partial sum for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += sA[ty][k] * sB[k][tx];
        }
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    if (A.size(1) != B.size(0)) {
        throw std::runtime_error("Incompatible matrix dimensions");
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matrix_mult_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matrix_mult_cpp_source = """
torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix multiplication
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return self.matrix_mult.matrix_mult_cuda(A, B)