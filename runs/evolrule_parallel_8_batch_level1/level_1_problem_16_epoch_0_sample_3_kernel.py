import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    int row = by + ty;
    int col = bx + tx;

    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    float acc = 0.0;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load A tile (A is K rows × M columns, so A^T is M × K)
        // Current A^T row is by + ty → column in A is (by + ty)
        // Current A^T column is k + tx → row in A is (k + tx)
        int A_col_in_A_transposed = k + tx;  // corresponds to row in A
        int A_row_in_A_transposed = by + ty; // corresponds to column in A

        if (A_col_in_A_transposed < K && A_row_in_A_transposed < M) {
            shared_A[ty][tx] = A[A_col_in_A_transposed * M + A_row_in_A_transposed];
        } else {
            shared_A[ty][tx] = 0.0;
        }

        // Load B tile (B is K × N)
        int B_row = k + ty;
        int B_col = bx + tx;
        if (B_row < K && B_col < N) {
            shared_B[ty][tx] = B[B_row * N + B_col];
        } else {
            shared_B[ty][tx] = 0.0;
        }

        __syncthreads();

        // Multiply and accumulate
        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += shared_A[ty][i] * shared_B[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        int index = row * N + col;
        C[index] = acc;
    }
}

torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure A and B are on the same device (GPU)
    A = A.cuda();
    B = B.cuda();

    // Dimensions
    int K = A.size(0); // Original A's rows (K)
    int M = A.size(1); // Original A's columns (M)
    int N = B.size(1); // B's columns (N)

    // Output tensor
    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    // Launch kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(),
                                                 B.data_ptr<float>(),
                                                 C.data_ptr<float>(),
                                                 M, K, N);

    return C;
}
"""

# Compile the CUDA code
custom_matmul = load_inline(
    name="custom_matmul",
    cuda_sources=matmul_source,
    functions=["custom_matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_matmul = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_matmul.custom_matmul_cuda(A, B)