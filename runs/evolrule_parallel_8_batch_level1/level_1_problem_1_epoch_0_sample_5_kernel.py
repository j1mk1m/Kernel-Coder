import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for optimized matrix multiplication using shared memory and tiling
matrixmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrixMulKernel(float *C, float *A, float *B, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockRow * TILE_WIDTH + ty;
    int col = blockCol * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // Load A tile into shared memory
        int aCol = m * TILE_WIDTH + tx;
        int aRow = row;
        if (aRow < N && aCol < N) {
            sA[ty][tx] = A[aRow * N + aCol];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        int bRow = m * TILE_WIDTH + ty;
        int bCol = col;
        if (bRow < N && bCol < N) {
            sB[ty][tx] = B[bRow * N + bCol];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the tiles
        for (int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write the result
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B, int N) {
    int block_size = TILE_WIDTH;
    dim3 dimBlock(block_size, block_size);
    int grid_size = (N + block_size - 1) / block_size;
    dim3 dimGrid(grid_size, grid_size);

    auto C = torch::empty({N, N}, A.options());

    matrixMulKernel<<<dimGrid, dimBlock>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        N
    );

    return C;
}
"""

matrixmul_cpp_source = "torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B, int N);"

# Load the CUDA extension
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        N = A.size(0)
        return matrixmul.matrixmul_cuda(A, B, N)

# Define input generation functions as in the original code
def get_inputs():
    A = torch.rand(4096, 4096).cuda()
    B = torch.rand(4096, 4096).cuda()
    return [A, B]

def get_init_inputs():
    return []