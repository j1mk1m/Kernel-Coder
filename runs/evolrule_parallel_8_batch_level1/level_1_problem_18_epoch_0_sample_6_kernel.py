import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel source code for matrix multiplication with transpose
matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

__global__ void matmul_transposed_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int blockRow = by * TILE_WIDTH;
    int blockCol = bx * TILE_WIDTH;
    
    int row = blockRow + ty;
    int col = blockCol + tx;
    
    float acc = 0.0f;
    
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load A_T[i][k] = A[k][i]
        int aCol = tile * TILE_WIDTH + tx;
        int aRow = row;
        if (aRow < M && aCol < K) {
            shared_A[ty][tx] = A[aCol * M + aRow];
        } else {
            shared_A[ty][tx] = 0.0f;
        }
        
        // Load B_T[k][j] = B[j][k]
        int bRow = tile * TILE_WIDTH + ty;
        int bCol = col;
        if (bRow < K && bCol < N) {
            shared_B[ty][tx] = B[bCol * K + bRow];
        } else {
            shared_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute the product of the current tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += shared_A[ty][k] * shared_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Kernel function for matrix multiplication with transpose
torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K) {
    auto device = A.device();
    auto a_data = A.data_ptr<float>();
    auto b_data = B.data_ptr<float>();
    
    auto C = torch::empty({M, N}, torch::device(device).dtype(torch::kFloat32));
    auto c_data = C.data_ptr<float>();
    
    const int block_size = TILE_WIDTH;
    dim3 block(block_size, block_size);
    
    int grid_rows = (M + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_cols = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 grid(grid_cols, grid_rows);
    
    matmul_transposed_kernel<<<grid, block>>>(a_data, b_data, c_data, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    
    return C;
}
"""

# Header for the matrix multiplication kernel
matmul_transposed_header = (
    "torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);"
)

# Compile the matrix multiplication kernel
matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_header,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
    extra_cuda_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

# Optional: Fused kernel with ReLU
matmul_transposed_relu_source = """
__global__ void matmul_transposed_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int blockRow = by * TILE_WIDTH;
    int blockCol = bx * TILE_WIDTH;
    
    int row = blockRow + ty;
    int col = blockCol + tx;
    
    float acc = 0.0f;
    
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int aCol = tile * TILE_WIDTH + tx;
        int aRow = row;
        if (aRow < M && aCol < K) {
            shared_A[ty][tx] = A[aCol * M + aRow];
        } else {
            shared_A[ty][tx] = 0.0f;
        }
        
        int bRow = tile * TILE_WIDTH + ty;
        int bCol = col;
        if (bRow < K && bCol < N) {
            shared_B[ty][tx] = B[bCol * K + bRow];
        } else {
            shared_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += shared_A[ty][k] * shared_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = fmaxf(acc, 0.0f);
    }
}

// Kernel function for fused matrix multiplication with ReLU
torch::Tensor matmul_transposed_relu_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K) {
    auto device = A.device();
    auto a_data = A.data_ptr<float>();
    auto b_data = B.data_ptr<float>();
    
    auto C = torch::empty({M, N}, torch::device(device).dtype(torch::kFloat32));
    auto c_data = C.data_ptr<float>();
    
    const int block_size = TILE_WIDTH;
    dim3 block(block_size, block_size);
    
    int grid_rows = (M + TILE_WIDTH - 1) / TILE_WIDTH;
    int grid_cols = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 grid(grid_cols, grid_rows);
    
    matmul_transposed_relu_kernel<<<grid, block>>>(a_data, b_data, c_data, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    
    return C;
}
"""

matmul_transposed_relu_header = (
    "torch::Tensor matmul_transposed_relu_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);"
)

# Compile the fused kernel (optional)
matmul_transposed_relu = load_inline(
    name="matmul_transposed_relu",
    cpp_sources=matmul_transposed_relu_header,
    cuda_sources=matmul_transposed_relu_source,
    functions=["matmul_transposed_relu_cuda"],
    verbose=True,
    extra_cuda_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_transposed = matmul_transposed
        self.matmul_transposed_relu = matmul_transposed_relu  # Optional fused kernel

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        K_A, M_A = A.shape
        N_B, K_B = B.shape
        assert K_A == K_B, "Dimension mismatch between A and B"
        
        M = M_A
        N = N_B
        K = K_A
        
        # Use fused ReLU kernel (optional) or standard kernel
        # return self.matmul_transposed_relu.matmul_transposed_relu_cuda(A, B, M, N, K)
        return self.matmul_transposed.matmul_transposed_cuda(A, B, M, N, K)