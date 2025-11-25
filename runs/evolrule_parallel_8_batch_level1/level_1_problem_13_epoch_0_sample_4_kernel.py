import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_symmetric_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void matmul_symmetric_tiled(const float* A, const float* B, float* C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;

    int row = blockRow + ty;
    int col = blockCol + tx;

    float Cvalue = 0.0f;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int m = 0; m < (N + TILE_DIM - 1) / TILE_DIM; ++m) {
        // Load A tile into shared memory
        int aRow = blockRow + ty;
        int aCol = m * TILE_DIM + tx;
        As[ty][tx] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;

        // Load B tile into shared memory
        int bRow = m * TILE_DIM + tx;
        int bCol = blockCol + ty;
        Bs[ty][tx] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        // Multiply the tiles and accumulate
        for (int k = 0; k < TILE_DIM; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matmul_symmetric_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    matmul_symmetric_tiled<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_symmetric_cpp_source = "torch::Tensor matmul_symmetric_cuda(torch::Tensor A, torch::Tensor B);"

matmul_symmetric = load_inline(
    name="matmul_symmetric",
    cpp_sources=matmul_symmetric_cpp_source,
    cuda_sources=matmul_symmetric_source,
    functions=["matmul_symmetric_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul_symmetric = matmul_symmetric

    def forward(self, A, B):
        return self.matmul_symmetric.matmul_symmetric_cuda(A, B)