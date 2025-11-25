import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication optimized for symmetric inputs
matmul_symm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_symm_kernel(const float* A, const float* B, float* C, int N) {
    int blockRow = blockIdx.y * TILE_WIDTH;
    int blockCol = blockIdx.x * TILE_WIDTH;
    int row = blockRow + threadIdx.y;
    int col = blockCol + threadIdx.x;
    
    float Cvalue = 0.0f;
    
    for (int k = 0; k < N; k += TILE_WIDTH) {
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
        
        // Load A's tile into shared memory (row-major)
        if (row < N && k + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + k + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B's tile into shared memory (transposed)
        if (k + threadIdx.y < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int n = 0; n < TILE_WIDTH; ++n) {
            Cvalue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matmul_symm_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are contiguous and square
    A = A.contiguous();
    B = B.contiguous();
    const int N = A.size(0);
    assert(A.size(0) == A.size(1) && B.size(0) == B.size(1));
    assert(A.size(0) == B.size(0)); // Both NxN
    
    auto C = torch::empty({N, N}, A.options());
    
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );
    
    matmul_symm_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    return C;
}
"""

matmul_symm_cpp_source = "torch::Tensor matmul_symm_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_symm = load_inline(
    name="matmul_symm",
    cpp_sources=matmul_symm_cpp_source,
    cuda_sources=matmul_symm_source,
    functions=["matmul_symm_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-Wno-deprecated-gpu-targets"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_symm = matmul_symm

    def forward(self, A, B):
        return self.matmul_symm.matmul_symm_cuda(A, B)