import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < M && col < N) {
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 16;
    dim3 grid_size((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
    dim3 block_size(block_size, block_size);

    matrix_mul_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matrix_mul_cpp_source = (
    "torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matrix_mul = load_inline(
    name="matrix_mul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mul = matrix_mul

    def forward(self, A, B):
        return self.matrix_mul.matrix_mul_cuda(A, B)