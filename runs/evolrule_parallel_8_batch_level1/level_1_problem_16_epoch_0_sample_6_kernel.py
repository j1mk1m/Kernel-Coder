import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TZ 16

__global__ void matmul_transposed_kernel(
    const float* a, const float* b, float* c,
    int M, int K, int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockRow = blockIdx.y * blockDim.y;
    int blockCol = blockIdx.x * blockDim.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TZ - 1) / TZ; tile++) {
        int kOffset = tile * TZ;

        // Load A tile (A^T's rows and columns)
        int aRow = blockRow + ty;
        int aCol = kOffset + tx;
        float aVal = 0.0f;
        if (aCol < K && aRow < M) {
            aVal = a[aCol * M + aRow]; // A[k][i] = A^T[i][k]
        }
        s_A[ty][tx] = aVal;

        // Load B tile (current K chunk and B's columns)
        int bRow = kOffset + tx;
        int bCol = blockCol + ty;
        float bVal = 0.0f;
        if (bRow < K && bCol < N) {
            bVal = b[bRow * N + bCol];
        }
        s_B[tx][ty] = bVal;

        __syncthreads();

        // Compute the partial sum for this tile
        for (int t = 0; t < TZ; ++t) {
            sum += s_A[ty][t] * s_B[t][tx];
        }

        __syncthreads();
    }

    // Write the result to the output
    int cRow = blockRow + ty;
    int cCol = blockCol + tx;
    if (cRow < M && cCol < N) {
        c[cRow * N + cCol] = sum;
    }
}

torch::Tensor matmul_transposed_cuda(
    torch::Tensor a, torch::Tensor b, int M, int K, int N) {
    assert(a.size(0) == K && a.size(1) == M);
    assert(b.size(0) == K && b.size(1) == N);

    auto output = torch::empty({M, N}, a.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    matmul_transposed_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        M, K, N
    );

    return output;
}
"""

matmul_transposed_cpp = """
torch::Tensor matmul_transposed_cuda(torch::Tensor a, torch::Tensor b, int M, int K, int N);
"""

matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_cpp,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_transposed_cuda = matmul_transposed.matmul_transposed_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda()
        B = B.cuda()
        M = A.size(1)
        K = A.size(0)
        N = B.size(1)
        return self.matmul_transposed_cuda(A, B, M, K, N)