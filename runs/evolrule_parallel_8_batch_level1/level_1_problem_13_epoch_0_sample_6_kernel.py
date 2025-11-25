import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication using shared memory
matrixmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrixMulKernel(float* C, const float* A, const float* B, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Csub = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // Compute global indices for A and B tiles
        int aRow = by * TILE_WIDTH + ty;
        int aCol = m * TILE_WIDTH + tx;

        int bRow = m * TILE_WIDTH + tx;
        int bCol = bx * TILE_WIDTH + ty;

        // Load data into shared memory
        As[ty][tx] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        // Multiply the tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write the result back
    int cRow = by * TILE_WIDTH + ty;
    int cCol = bx * TILE_WIDTH + tx;
    if (cRow < N && cCol < N) {
        C[cRow * N + cCol] = Csub;
    }
}

torch::Tensor matrixMul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    assert(A.size(0) == N && A.size(1) == N);
    assert(B.size(0) == N && B.size(1) == N);

    auto C = torch::empty({N, N}, torch::device(A.device()));
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    int grid_dim = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 blocks(grid_dim, grid_dim);

    matrixMulKernel<<<blocks, threads>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);

    return C;
}
"""

matrixmul_cpp_source = "torch::Tensor matrixMul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixMul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixmul = matrixmul

    def forward(self, A, B):
        return self.matrixmul.matrixMul_cuda(A, B)

N = 4096

def get_inputs():
    A = torch.rand(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.rand(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A.cuda(), B.cuda()]  # Move tensors to GPU

def get_init_inputs():
    return []