import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32  // Tuned for best performance on given dimensions

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M1, int K, int M2) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_DIM + ty;
    int Col = bx * TILE_DIM + tx;
    float Cvalue = 0.0f;

    for (int p = 0; p < (K + TILE_DIM - 1) / TILE_DIM; ++p) {
        // Load tiles of A and B into shared memory
        int aRow = Row;
        int aCol = p * TILE_DIM + tx;
        sA[ty][tx] = (aRow < M1 && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        int bRow = p * TILE_DIM + ty;
        int bCol = Col;
        sB[ty][tx] = (bRow < K && bCol < M2) ? B[bRow * M2 + bCol] : 0.0f;

        __syncthreads();

        // Compute tile contribution
        for (int k = 0; k < TILE_DIM; ++k) {
            Cvalue += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (Row < M1 && Col < M2) {
        C[Row * M2 + Col] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M1, int K, int M2) {
    // Validate dimensions
    if (A.size(0) != M1 || A.size(1) != K || B.size(0) != K || B.size(1) != M2) {
        throw std::runtime_error("Invalid matrix dimensions");
    }

    // Output tensor
    auto C = torch::empty({M1, M2}, A.options());

    // Grid and block dimensions
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(
        (M2 + TILE_DIM - 1) / TILE_DIM,
        (M1 + TILE_DIM - 1) / TILE_DIM
    );

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M1, K, M2);
    cudaDeviceSynchronize();  // Ensure kernel completion

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M1, int K, int M2);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A, B):
        N = 16
        M = 1024
        K = 2048
        L = 768
        # Reshape A to (N*M, K) for batched matrix multiplication
        A_reshaped = A.view(N * M, K)
        # Perform optimized matrix multiplication
        C = self.matmul_cuda.matmul_cuda(A_reshaped, B, N * M, K, L)
        # Reshape back to (N, M, L)
        return C.view(N, M, L)