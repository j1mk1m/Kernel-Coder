import torch
import torch.nn as nn
from custom_cuda_ops import create_cuda_op

cpp_source = """
#include <torch/extension.h>

torch::Tensor matrixMul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int TileWidth>
__global__ void matrixMulKernel(float* C, const float* A, const float* B,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Row = by * TileWidth;
    int Col = bx * TileWidth;

    int RowInBlock = threadIdx.y;
    int ColInBlock = threadIdx.x;

    float Cvalue = 0.0f;

    for (int m = 0; m < (numAColumns - 1) / TileWidth + 1; ++m) {
        __shared__ float As[TileWidth][TileWidth];
        __shared__ float Bs[TileWidth][TileWidth];

        int aRow = Row + RowInBlock;
        int aCol = m * TileWidth + ColInBlock;
        As[RowInBlock][ColInBlock] = (aRow < numARows && aCol < numAColumns) ?
            A[aRow * numAColumns + aCol] : 0.0f;

        int bRow = m * TileWidth + RowInBlock;
        int bCol = Col + ColInBlock;
        Bs[RowInBlock][ColInBlock] = (bRow < numBRows && bCol < numBColumns) ?
            B[bRow * numBColumns + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TileWidth; ++k) {
            Cvalue += As[RowInBlock][k] * Bs[k][ColInBlock];
        }

        __syncthreads();
    }

    int cRow = Row + RowInBlock;
    int cCol = Col + ColInBlock;
    if (cRow < numCRows && cCol < numCColumns) {
        C[cRow * numCColumns + cCol] = Cvalue;
    }
}

torch::Tensor matrixMul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N) {
    auto A_t = A.t().contiguous();
    auto B_t = B.t().contiguous();

    auto C = torch::empty({M, N}, A.options());

    const int TileWidth = 16;
    dim3 threads(TileWidth, TileWidth);
    dim3 blocks(
        (N + TileWidth - 1) / TileWidth,
        (M + TileWidth - 1) / TileWidth
    );

    matrixMulKernel<TileWidth><<<blocks, threads>>>(
        C.data_ptr<float>(),
        A_t.data_ptr<float>(),
        B_t.data_ptr<float>(),
        M, K,
        K, N,
        M, N
    );

    return C;
}
"""

matrixmul_op = create_cuda_op(cpp_source, cuda_source, ["matrixMul_cuda"], "matrixmul")

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixmul = matrixmul_op

    def forward(self, A, B):
        M = A.size(1)
        K = A.size(0)
        N = B.size(0)
        return self.matrixmul.matrixMul_cuda(A, B, M, K, N)