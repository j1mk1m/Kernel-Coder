import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K) {
    // Shared memory declarations
    extern __shared__ float s_A[];
    extern __shared__ float s_B[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Load data into shared memory
    if (row < M && threadIdx.x < K) {
        s_A[threadIdx.x] = A[row * K + threadIdx.x];
    } else {
        s_A[threadIdx.x] = 0.0f;
    }

    if (col < K && threadIdx.y < M) {
        s_B[threadIdx.y] = B[col * M + threadIdx.y];
    } else {
        s_B[threadIdx.y] = 0.0f;
    }

    __syncthreads();

    // Perform matrix multiplication
    for (int k = 0; k < K; ++k) {
        sum += s_A[k] * s_B[k];
    }

    __syncthreads();

    // Store result in global memory
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t sharedMemSize = 2 * sizeof(float) * threadsPerBlock.x * threadsPerBlock.y;

    matmul_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix-vector multiplication
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
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)