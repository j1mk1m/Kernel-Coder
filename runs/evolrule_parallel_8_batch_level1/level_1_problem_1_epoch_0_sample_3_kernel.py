import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix multiplication
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_mult_kernel(const float* A, const float* B, float* C, 
                                  int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;
    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    auto C = torch::empty({N, N}, A.options());

    matrix_mult_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), 
        C.data_ptr<float>(), N
    );

    cudaDeviceSynchronize();
    return C;
}
"""

matrix_mult_cpp_source = (
    "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B, int N);"
)

# Compile the inline CUDA code
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.N = 2048 * 2
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device
        A = A.contiguous()
        B = B.contiguous()
        return self.matrix_mult.matrix_mult_cuda(A, B, self.N)