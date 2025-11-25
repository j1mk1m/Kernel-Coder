import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrixmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TB 32

__global__ void matrixMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float A_shrd[TB][TB];
    __shared__ float B_shrd[TB][TB];

    int c_row = blockIdx.y * TB + threadIdx.y;
    int c_col = blockIdx.x * TB + threadIdx.x;

    if (c_row >= M || c_col >= N) return;

    float sum = 0.0f;

    for (int t = 0; t < (K + TB - 1) / TB; t++) {
        // Load A tile into shared memory
        int a_col = t * TB + threadIdx.x;
        int a_row = c_row;
        if (a_col < K) {
            A_shrd[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            A_shrd[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile into shared memory
        int b_row = t * TB + threadIdx.y;
        int b_col = c_col + threadIdx.x;
        if (b_row < K) {
            B_shrd[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            B_shrd[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < TB; k++) {
            sum += A_shrd[threadIdx.y][k] * B_shrd[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[c_row * N + c_col] = sum;
}

torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TB, TB);
    dim3 blocks(
        (N + TB - 1) / TB,
        (M + TB - 1) / TB
    );

    matrixMulKernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matrixmul_cpp_source = """
torch::Tensor matrixmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA code
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrixmul.matrixmul_cuda(A, B)