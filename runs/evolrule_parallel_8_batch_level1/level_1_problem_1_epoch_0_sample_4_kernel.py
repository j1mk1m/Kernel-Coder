import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrix_mult_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cvalue = 0.0;

    for (int m = 0; m < (N + TILE_WIDTH - 1)/TILE_WIDTH; m++) {
        // Load the current tile of A and B into shared memory
        if (by * TILE_WIDTH + ty < N && m * TILE_WIDTH + tx < N) {
            shared_A[ty][tx] = A[(by * TILE_WIDTH + ty) * N + (m * TILE_WIDTH + tx)];
        } else {
            shared_A[ty][tx] = 0.0;
        }

        if (m * TILE_WIDTH + ty < N && bx * TILE_WIDTH + tx < N) {
            shared_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + (bx * TILE_WIDTH + tx)];
        } else {
            shared_B[ty][tx] = 0.0;
        }

        __syncthreads();

        // Perform the multiplication for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    // Write the computed value to the output matrix
    if (by * TILE_WIDTH + ty < N && bx * TILE_WIDTH + tx < N) {
        int row = by * TILE_WIDTH + ty;
        int col = bx * TILE_WIDTH + tx;
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid((N + block.x - 1)/block.x, (N + block.y - 1)/block.y);

    auto C = torch::empty({N, N}, A.options());

    matrix_mult_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

matrix_mult_cpp_source = (
    "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A, B)