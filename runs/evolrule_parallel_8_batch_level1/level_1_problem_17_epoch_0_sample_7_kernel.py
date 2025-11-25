import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define M 2048
#define N 4096
#define K 8192
#define TILE_WIDTH 16

__global__ void matmul_kernel(float *A, float *B, float *C) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x * TILE_WIDTH;
    int by = blockIdx.y * TILE_WIDTH;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load shared_A with A's portion (rows in block, columns in current tile)
        int a_row = by + ty;
        int a_col = t * TILE_WIDTH + tx;
        if (a_row < M && a_col < K) {
            shared_A[ty][tx] = A[a_row * K + a_col];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        // Load shared_B with B's portion (columns in block, rows in current tile)
        int b_row = bx + tx; // B's row (column in B^T)
        int b_col = t * TILE_WIDTH + ty; // B's column (row in B^T)
        if (b_row < N && b_col < K) {
            shared_B[ty][tx] = B[b_row * K + b_col]; // B[j][k]
        } else {
            shared_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum using the current tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    // Write result to output matrix
    int row = by + ty;
    int col = bx + tx;
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto output = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>());
    cudaDeviceSynchronize();

    return output;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the custom CUDA kernel
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)