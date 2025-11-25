import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix multiplication with transpose optimization
matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define block size
#define BLOCK_SIZE 32

__global__ void matmul_transpose_kernel(
    const float* __restrict__ A, const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N, int K) {

    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for tile blocks
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0;

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {

        // Load A tile into shared memory
        int a_row = row;
        int a_col = tile * BLOCK_SIZE + threadIdx.x;
        bool a_in_bounds = a_col < K;
        shared_A[threadIdx.y][threadIdx.x] = 
            (a_in_bounds && a_row < M) ? A[a_row * K + a_col] : 0.0f;
        
        // Load B tile into shared memory (note B is transposed in the computation)
        int b_col = col;
        int b_row = tile * BLOCK_SIZE + threadIdx.y;
        bool b_in_bounds = b_row < K;
        shared_B[threadIdx.y][threadIdx.x] = 
            (b_in_bounds && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        // Synchronize to make sure the shared memory is loaded
        __syncthreads();

        // Compute the dot product of the current tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        // Synchronize to proceed to next tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(
    torch::Tensor A, torch::Tensor B, int M, int N, int K) {

    // Output tensor
    auto C = torch::empty({M, N}, A.options());

    // Define block and grid dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // Launch kernel
    matmul_transpose_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), 
        C.data_ptr<float>(), M, N, K);

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_transpose_cpp_source = (
    "torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);"
)

# Compile the CUDA code
matmul_transpose = load_inline(
    name="matmul_transpose",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transpose = matmul_transpose

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Extract shapes (assuming A is MxK, B is NxK)
        M = A.size(0)
        N = B.size(0)
        K = A.size(1)
        return self.matmul_transpose.matmul_transpose_cuda(A, B, M, N, K)