import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with a diagonal matrix
matrix_mul_diag_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_mul_diag_kernel(const float* A, const float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[i] * B[row][col];
        }
        C[row][col] = sum;
    }
}

torch::Tensor matrix_mul_diag_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = B.size(1);
    auto C = torch::zeros({N, M}, A.options());

    const int block_size = 32;
    const int grid_x = (M + block_size - 1) / block_size;
    const int grid_y = (N + block_size - 1) / block_size;

    matrix_mul_diag_kernel<<<grid_x, grid_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

    return C;
}
"""

matrix_mul_diag_cpp_source = (
    "torch::Tensor matrix_mul_diag_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication with a diagonal matrix
matrix_mul_diag = load_inline(
    name="matrix_mul_diag",
    cpp_sources=matrix_mul_diag_cpp_source,
    cuda_sources=matrix_mul_diag_source,
    functions=["matrix_mul_diag_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mul_diag = matrix_mul_diag

    def forward(self, A, B):
        return self.matrix_mul_diag.matrix_mul_diag_cuda(A, B)