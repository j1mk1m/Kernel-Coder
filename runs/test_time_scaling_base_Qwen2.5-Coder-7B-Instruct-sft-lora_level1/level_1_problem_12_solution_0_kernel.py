import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with a diagonal matrix
matmul_diag_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_diag_kernel(const float* A, const float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        C[row * M + col] = 0.0f;
        for (int k = 0; k < N; ++k) {
            C[row * M + col] += A[k] * B[k * M + col];
        }
    }
}

torch::Tensor matmul_diag_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = B.size(1);
    auto C = torch::zeros({N, M}, A.options());

    const int block_size = 32;
    const int num_blocks_x = (M + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;

    matmul_diag_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

    return C;
}
"""

matmul_diag_cpp_source = (
    "torch::Tensor matmul_diag_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication with a diagonal matrix
matmul_diag = load_inline(
    name="matmul_diag",
    cpp_sources=matmul_diag_cpp_source,
    cuda_sources=matmul_diag_source,
    functions=["matmul_diag_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return matmul_diag.matmul_diag_cuda(A, B)


def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]


def get_init_inputs():
    return []