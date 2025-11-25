import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

matmul_transposed_tiled_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TS 16

__global__ void matmul_transposed_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sA[TS][TS];
    __shared__ float sB[TS][TS];

    int row = by * TS + ty;
    int col = bx * TS + tx;

    float psum = 0.0;

    for (int m = 0; m < (K + TS - 1) / TS; m++) {
        // Load A tile into shared memory
        int aRow = by * TS + ty;
        int aCol = m * TS + tx;
        if (aRow < M && aCol < K)
            sA[ty][tx] = A[aRow * K + aCol];
        else
            sA[ty][tx] = 0.0;

        // Load B tile into shared memory (B is NxK)
        int bRow = bx * TS + tx;
        int bCol = m * TS + ty;
        if (bRow < N && bCol < K)
            sB[ty][tx] = B[bRow * K + bCol];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < TS; ++k) {
            psum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = psum;
    }
}

torch::Tensor matmul_transposed_tiled_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    assert(B.size(1) == K);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TS, TS);
    dim3 blocks((N + TS - 1) / TS, (M + TS - 1) / TS);

    matmul_transposed_tiled<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

cpp_source = """
extern "C" {
    torch::Tensor matmul_transposed_tiled_cuda(torch::Tensor, torch::Tensor);
}
"""

matmul_transposed_tiled = load_inline(
    name="matmul_transposed_tiled",
    cpp_sources=[cpp_source],
    cuda_sources=[matmul_transposed_tiled_source],
    functions=["matmul_transposed_tiled_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_transposed_tiled = matmul_transposed_tiled

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_transposed_tiled.matmul_transposed_tiled_cuda(A, B)

def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []