import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication and Mish activation
matrix_multiplication_and_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_device(float x) {
    return x * tanh(log(1 + exp(x)));
}

__global__ void matrix_multiplication_and_mish_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = mish_device(sum);
    }
}

torch::Tensor matrix_multiplication_and_mish_cuda(torch::Tensor A, torch::Tensor B) {
    auto m = A.size(0);
    auto n = B.size(1);
    auto k = A.size(1);
    auto C = torch::zeros({m, n}, A.options());

    const int block_size = 16;
    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    matrix_multiplication_and_mish_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), m, n, k);

    return C;
}
"""

matrix_multiplication_and_mish_cpp_source = (
    "torch::Tensor matrix_multiplication_and_mish_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication and Mish activation
matrix_multiplication_and_mish = load_inline(
    name="matrix_multiplication_and_mish",
    cpp_sources=matrix_multiplication_and_mish_cpp_source,
    cuda_sources=matrix_multiplication_and_mish_source,
    functions=["matrix_multiplication_and_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.matrix_multiplication_and_mish = matrix_multiplication_and_mish

    def forward(self, x):
        x = self.matrix_multiplication_and_mish_matrix_multiplication_and_mish_cuda(x, self.linear.weight.t())
        x = torch.nn.functional.mish(x)
        return x