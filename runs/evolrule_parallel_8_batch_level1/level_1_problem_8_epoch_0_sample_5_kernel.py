import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized matrix multiplication with irregular dimensions
matrixmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrixmul_kernel(const float* A, const float* B, float* C, 
                                int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float C_value = 0.0;

    if (row < M && col < N) {
        for (int k = 0; k < K; k += TILE_WIDTH) {
            __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
            __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

            int a_row = row;
            int a_col = k + threadIdx.x;
            if (a_col < K) {
                shared_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
            } else {
                shared_A[threadIdx.y][threadIdx.x] = 0.0;
            }

            int b_row = k + threadIdx.y;
            int b_col = col;
            if (b_row < K) {
                shared_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
            } else {
                shared_B[threadIdx.y][threadIdx.x] = 0.0;
            }

            __syncthreads();

            for (int t = 0; t < TILE_WIDTH; ++t) {
                C_value += shared_A[threadIdx.y][t] * shared_B[t][threadIdx.x];
            }

            __syncthreads();
        }
        C[row * N + col] = C_value;
    }
}

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    matrixmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), 
                                         B.data_ptr<float>(), 
                                         C.data_ptr<float>(), 
                                         M, K, N);

    return C;
}
"""

matrixmul_cpp_source = (
    "torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrixmul.matrixmul_cuda(A, B)

# Maintaining the original input generation for compatibility
M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []