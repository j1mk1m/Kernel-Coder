import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
tensor_matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor_matrix_mult_kernel(const float* A, const float* B, float* C, int N, int M, int K, int L) {
    int n = blockIdx.x;
    int m = blockIdx.y * blockDim.x + threadIdx.x;
    int l = blockIdx.z * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[n * M * K + m * K + k] * B[k * L + l];
    }

    C[n * M * L + m * L + l] = sum;
}

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = A.size(1);
    auto K = A.size(2);
    auto L = B.size(1);

    auto C = torch::zeros({N, M, L}, A.options());

    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid((M + threads_per_block.x - 1) / threads_per_block.x,
                          (L + threads_per_block.y - 1) / threads_per_block.y,
                          N);

    tensor_matrix_mult_kernel<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K, L);

    return C;
}
"""

tensor_matrix_mult_cpp_source = (
    "torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for 3D tensor-matrix multiplication
tensor_matrix_mult = load_inline(
    name="tensor_matrix_mult",
    cpp_sources=tensor_matrix_mult_cpp_source,
    cuda_sources=tensor_matrix_mult_source,
    functions=["tensor_matrix_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matrix_mult = tensor_matrix_mult

    def forward(self, A, B):
        return self.tensor_matrix_mult.tensor_matrix_mult_cuda(A, B)