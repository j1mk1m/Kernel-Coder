import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrixmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrixMulKernel(float* C, float* A, float* B, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        int aRow = Row;
        int aCol = m * TILE_WIDTH + tx;
        int bRow = m * TILE_WIDTH + ty;
        int bCol = Col;

        if (aCol < N && aRow < N)
            As[ty][tx] = A[aRow * N + aCol];
        else
            As[ty][tx] = 0.0f;

        if (bRow < N && bCol < N)
            Bs[ty][tx] = B[bRow * N + bCol];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

torch::Tensor matrixMulCUDA(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    assert(A.size(1) == N && B.size(0) == N && B.size(1) == N);

    auto options = A.options();
    torch::Tensor C = torch::zeros({N, N}, options);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matrixMulKernel<<<blocks, threads>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}
"""

matrixmul_cpp = "torch::Tensor matrixMulCUDA(torch::Tensor A, torch::Tensor B);"

# Compile the CUDA kernel
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp,
    cuda_sources=matrixmul_source,
    functions=["matrixMulCUDA"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrixmul.matrixMulCUDA(A, B)