import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block's starting indices
    int block_start_row = by * TILE_SIZE;
    int block_start_col = bx * TILE_SIZE;

    // Thread's position within the block
    int row_in_block = ty;
    int col_in_block = tx;

    // Shared memory for tiles of A and B
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load tile of A into shared memory
        int a_row = block_start_row + row_in_block;
        int a_col = m * TILE_SIZE + col_in_block;
        if (a_row < N && a_col < N) {
            shared_A[row_in_block][col_in_block] = A[a_row * N + a_col];
        } else {
            shared_A[row_in_block][col_in_block] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = m * TILE_SIZE + row_in_block;
        int b_col = block_start_col + col_in_block;
        if (b_row < N && b_col < N) {
            shared_B[row_in_block][col_in_block] = B[b_row * N + b_col];
        } else {
            shared_B[row_in_block][col_in_block] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[row_in_block][k] * shared_B[k][col_in_block];
        }

        __syncthreads();
    }

    // Write the result
    int global_row = block_start_row + row_in_block;
    int global_col = block_start_col + col_in_block;
    if (global_row < N && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    assert(A.size(0) == A.size(1) && B.size(0) == B.size(1));
    assert(A.size(1) == B.size(0));

    auto C = torch::empty({N, N}, torch::device("cuda"));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul = load_inline(
    name="matmul",
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
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)

# Adjust get_inputs to generate CUDA tensors
N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []