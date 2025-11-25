import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrixmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matrixMulKernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // Thread index within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Row and column of the C element computed by the thread
    int row = blockRow * TILE_SIZE + ty;
    int col = blockCol * TILE_SIZE + tx;

    float Cvalue = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Shared memory for the tiles of A and B
        __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
        __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

        // Compute indices for A and B
        int aRow = row;
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;
        int bCol = col;

        // Load data from global to shared memory
        if (aCol < K && aRow < M) {
            sharedA[ty][tx] = A[aRow * K + aCol];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (bCol < N && bRow < K) {
            sharedB[ty][tx] = B[bRow * N + bCol];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product of the loaded tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += sharedA[ty][k] * sharedB[k][tx];
        }
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matrixmul_cpp_source = "torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B);"

matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixmul = matrixmul

    def forward(self, A, B):
        return self.matrixmul.matrixmul_cuda(A, B)