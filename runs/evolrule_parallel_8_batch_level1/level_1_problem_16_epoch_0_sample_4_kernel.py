import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

template <int BLOCK_SIZE>
__global__ void matmul_transposed(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int K, int M, int N,
    int lda_A, int ldb_B, int ldc_C
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockRow = blockIdx.y * BLOCK_SIZE;
    int blockCol = blockIdx.x * BLOCK_SIZE;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float Cval = 0.0f;

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load A tile (A^T dimensions)
        int aRow = blockRow + ty;
        int aCol = t * BLOCK_SIZE + tx;
        int A_offset = aCol * lda_A + aRow;
        As[ty][tx] = (aCol < K && aRow < M) ? A[A_offset] : 0.0f;

        // Load B tile
        int bRow = t * BLOCK_SIZE + ty;
        int bCol = blockCol + tx;
        int B_offset = bRow * ldb_B + bCol;
        Bs[ty][tx] = (bRow < K && bCol < N) ? B[B_offset] : 0.0f;

        __syncthreads();

        // Perform partial computation
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            Cval += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write result
    int cRow = blockRow + ty;
    int cCol = blockCol + tx;
    if (cRow < M && cCol < N) {
        int C_offset = cRow * ldc_C + cCol;
        C[C_offset] = Cval;
    }
}

void matmul_transposed_cuda(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int K, int M, int N
) {
    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    matmul_transposed<BLOCK_SIZE><<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        K, M, N,
        A.stride(0),
        B.stride(0),
        C.stride(0)
    );
}
"""

matmul_transposed_header = """
void matmul_transposed_cuda(
    torch::Tensor A, torch::Tensor B, torch::Tensor C,
    int K, int M, int N
);
"""

matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_header,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.K = 4096 * 2
        self.M = 1024 * 2
        self.N = 2048 * 2

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()
        C = torch.empty((self.M, self.N), device=A.device, dtype=A.dtype)
        matmul_transposed.matmul_transposed_cuda(A, B, C, self.K, self.M, self.N)
        return C