import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_3d_tiled_source = """
#include <torch/extension.h>

__global__ void matmul_3d_tiled(
    const float* A, const float* B, float* C,
    int N, int M, int K, int L, int n) {
    const int TILE_SIZE = 16;
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    float sum = 0.0f;

    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load tile from A into sA
        int a_row = by + ty;
        int a_col = m * TILE_SIZE + tx;
        sA[ty][tx] = (a_col < K && a_row < M) ?
            A[n * M * K + a_row * K + a_col] : 0.0f;

        // Load tile from B into sB
        int b_row = m * TILE_SIZE + tx;
        int b_col = bx + ty;
        sB[ty][tx] = (b_row < K && b_col < L) ?
            B[b_row * L + b_col] : 0.0f;

        __syncthreads();

        // Perform element-wise multiplication and accumulation
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    int row = by + ty;
    int col = bx + tx;
    if (row < M && col < L) {
        C[n * M * L + row * L + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    assert(A.size(2) == B.size(0), "Incompatible dimensions");

    auto C = torch::empty({N, M, L}, A.options());

    const int TILE_SIZE = 16;
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    int blocks_per_grid_x = (L + TILE_SIZE - 1) / TILE_SIZE;
    int blocks_per_grid_y = (M + TILE_SIZE - 1) / TILE_SIZE;
    dim3 blocks_per_grid(blocks_per_grid_x, blocks_per_grid_y);

    for (int n = 0; n < N; ++n) {
        matmul_3d_tiled<<<blocks_per_grid, threads_per_block>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            N, M, K, L, n
        );
    }

    return C;
}
"""

matmul_3d_tiled_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda_op = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_3d_tiled_cpp_source,
    cuda_sources=matmul_3d_tiled_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda_op

    def forward(self, A, B):
        return self.matmul_cuda(A, B)