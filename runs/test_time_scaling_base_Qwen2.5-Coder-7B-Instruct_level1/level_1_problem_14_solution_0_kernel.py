import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication of upper triangular matrices
upper_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k <= min(row, col); ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto n = A.size(0);
    auto C = torch::zeros({n, n}, A.options());

    const int block_size = 32;
    const int grid_rows = (n + block_size - 1) / block_size;
    const int grid_cols = (n + block_size - 1) / block_size;

    upper_triangular_matmul_kernel<<<grid_rows, grid_cols, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);

    return C;
}
"""

upper_triangular_matmul_cpp_source = (
    "torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix