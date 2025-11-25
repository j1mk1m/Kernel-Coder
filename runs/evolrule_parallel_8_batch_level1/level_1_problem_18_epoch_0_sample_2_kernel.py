import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for tiled matrix multiplication of transposed matrices
matmul_tiled_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_transposed_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Pvalue = 0.0f;

    for (int p = 0; p < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        // Load A block into shared memory
        int aRow = by * TILE_WIDTH + ty; // row in A^T (column of A)
        int aCol = p * TILE_WIDTH + tx; // column in A^T (row of A)
        bool a_in_bounds = (aCol < K) && (aRow < M);
        sA[ty][tx] = a_in_bounds ? A[aCol * M + aRow] : 0.0f;

        // Load B block into shared memory
        int bRow = p * TILE_WIDTH + ty; // row in B^T (column of B)
        int bCol = bx * TILE_WIDTH + tx; // column in B^T (row of B)
        bool b_in_bounds = (bRow < K) && (bCol < N);
        sB[ty][tx] = b_in_bounds ? B[bCol * K + bRow] : 0.0f;

        __syncthreads();

        // Perform the multiplication
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write the result to global memory
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}

torch::Tensor matmul_transposed_tiled_cuda(
    torch::Tensor A, torch::Tensor B, int M, int N, int K
) {
    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matmul_transposed_tiled<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

matmul_tiled_cpp_source = (
    "torch::Tensor matmul_transposed_tiled_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);"
)

# Compile the inline CUDA code for matrix multiplication of transposed matrices
matmul_tiled = load_inline(
    name="matmul_tiled",
    cpp_sources=matmul_tiled_cpp_source,
    cuda_sources=matmul_tiled_source,
    functions=["matmul_transposed_tiled_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_tiled = matmul_tiled

    def forward(self, A, B):
        M = A.size(1)
        N = B.size(0)
        K = A.size(0)
        return self.matmul_tiled.matmul_transposed_tiled_cuda(A, B, M, N, K)