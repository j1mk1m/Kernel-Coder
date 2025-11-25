import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrixmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 32  // Tile size for shared memory (adjust based on GPU architecture)

template<typename T>
__global__ void matrixmul_kernel(const T* A, const T* B, T* C, 
                                int M, int K, int N) {
    // Block and thread indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Block's starting indices in the output matrix
    int Row = by * TILE_DIM + ty;
    int Col = bx * TILE_DIM + tx;

    T val = 0;
    // Each tile covers a TILE_DIM x TILE_DIM block of the output
    for (int p = 0; p < (K + TILE_DIM - 1) / TILE_DIM; ++p) {
        // Shared memory for tiles of A and B
        __shared__ T As[TILE_DIM][TILE_DIM];
        __shared__ T Bs[TILE_DIM][TILE_DIM];

        int aRow = Row;
        int aCol = p * TILE_DIM + tx;
        int bRow = p * TILE_DIM + ty;
        int bCol = Col;

        // Load tiles into shared memory
        As[ty][tx] = (aCol < K && aRow < M) ? A[aRow * K + aCol] : 0;
        Bs[ty][tx] = (bCol < N && bRow < K) ? B[bRow * N + bCol] : 0;

        __syncthreads();

        // Compute the dot product of the current tiles
        for (int k = 0; k < TILE_DIM; ++k) {
            val += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    // Write the result to global memory if within bounds
    if (Row < M && Col < N) {
        C[Row * N + Col] = val;
    }
}

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // Grid and block dimensions
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1)/TILE_DIM, (M + TILE_DIM - 1)/TILE_DIM);

    // Launch kernel with template specialization for float
    matrixmul_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), 
        C.data_ptr<float>(), M, K, N
    );

    return C;
}
"""

matrixmul_cpp_source = (
    "torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrixmul.matrixmul_cuda(A, B)