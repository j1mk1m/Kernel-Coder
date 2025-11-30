import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
matrix_scalar_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_scalar_multiplication_kernel(const float* A, float s, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] * s;
    }
}

torch::Tensor matrix_scalar_multiplication_cuda(torch::Tensor A, float s) {
    auto M = A.size(0);
    auto N = A.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

    matrix_scalar_multiplication_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), M, N);

    return C;
}
"""

matrix_scalar_multiplication_cpp_source = (
    "torch::Tensor matrix_scalar_multiplication_cuda(torch::Tensor A, float s);"
)

# Compile the inline CUDA code for matrix-scalar multiplication
matrix_scalar_multiplication = load_inline(
    name="matrix_scalar_multiplication",
    cpp_sources=matrix_scalar_multiplication_cpp_source,
    cuda_sources=matrix_scalar_multiplication_source,
    functions=["matrix_scalar_multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super(ModelNew, self).__init__()
        self.matrix_scalar_multiplication = matrix_scalar_multiplication
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.matrix_scalar_multiplication.matrix_scalar_multiplication_cuda(A, s)