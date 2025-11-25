import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_transposed_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {

    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Indices in C
    int i = by * TILE_WIDTH + ty;
    int j = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        int k_start = m * TILE_WIDTH;
        int k = k_start + tx;

        // Load tiles into shared memory
        if (k < K) {
            s_A[ty][tx] = A[k * M + i];  // A is (K, M)
            s_B[ty][tx] = B[k * N + j];  // B is (K, N)
        } else {
            s_A[ty][tx] = 0.0f;
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product for this tile
        for (int k_local = 0; k_local < TILE_WIDTH; ++k_local) {
            Cvalue += s_A[ty][k_local] * s_B[tx][k_local];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (i < M && j < N) {
        C[i * N + j] = Cvalue;
    }
}

torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B) {
    // Extract dimensions
    int M = A.size(1); // A is (K, M)
    int K = A.size(0);
    int N = B.size(1); // B is (K, N)

    // Output tensor (M x N)
    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int grid_x = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_y = (M + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 grid(grid_x, grid_y);

    // Launch kernel
    matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, K, N
    );

    return C;
}
"""

matmul_transposed_cpp_source = (
    "torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_cpp_source,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_transposed = matmul_transposed

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_transposed.matmul_transposed_cuda(A, B)