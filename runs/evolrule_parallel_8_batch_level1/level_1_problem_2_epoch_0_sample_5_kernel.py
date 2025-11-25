import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrixmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixmul_kernel(float* d_C, const float* d_A, const float* d_B, 
                                int M, int K, int N) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread index within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Shared memory for the tiles of A and B
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    float Csub = 0.0;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load the current tile of A into shared memory
        int aRow = blockRow * TILE_WIDTH + row;
        int aCol = m * TILE_WIDTH + col;
        s_A[row][col] = (aRow < M && aCol < K) ? 
                        d_A[aRow * K + aCol] : 0.0f;

        // Load the current tile of B into shared memory (transposed)
        int bRow = m * TILE_WIDTH + row;
        int bCol = blockCol * TILE_WIDTH + col;
        s_B[row][col] = (bRow < K && bCol < N) ? 
                        d_B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        // Compute the products for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Csub += s_A[row][k] * s_B[k][col];
        }

        __syncthreads();
    }

    // Write the result to global memory
    int cRow = blockRow * TILE_WIDTH + row;
    int cCol = blockCol * TILE_WIDTH + col;
    if (cRow < M && cCol < N) {
        d_C[cRow * N + cCol] = Csub;
    }
}

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Calculate grid and block dimensions
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Launch the kernel
    matrixmul_kernel<<<blocks, threads>>>(C.data_ptr<float>(),
                                         A.data_ptr<float>(),
                                         B.data_ptr<float>(),
                                         M, K, N);

    return C;
}
"""

# Compile the inline CUDA code
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=[""],
    cuda_sources=matrixmul_source,
    functions=["matrixmul_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrixmul.matrixmul_cuda(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    # Generate tensors on CUDA
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []