import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with fused group normalization
fused_group_norm_matrix_multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_group_norm_matrix_multiplication_kernel(const float* A, const float* B, const float* gamma, const float* beta, float* C, int N, int G) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        float mean = sum / N;
        float variance = 0.0f;
        for (int k = 0; k < N; ++k) {
            variance += pow(sum - mean, 2);
        }
        variance /= N;
        C[row * N + col] = gamma[g] * (sum - mean) / sqrt(variance + 1e-5) + beta[g];
    }
}

torch::Tensor fused_group_norm_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor gamma, torch::Tensor beta) {
    auto N = A.size(0);
    auto G = A.size(1) / A.size(2);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads_per_block(32, 32, 1);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x, (N + threads_per_block.y - 1) / threads_per_block.y, (G + threads_per_block.z - 1) / threads_per_block.z);

    fused_group_norm_matrix_multiplication_kernel<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), C.data_ptr<float>(), N, G);

    return C;
}
"""

fused_group_norm_matrix_multiplication_cpp_source = (
    "torch::Tensor fused_group_norm_matrix_multiplication_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor gamma, torch::Tensor beta);"
)

# Compile the inline CUDA code for fused group normalization matrix multiplication
fused_group_norm_matrix_multiplication = load_inline(
    name="fused_group_norm_matrix_multiplication",
    cpp_sources=fused_group_norm_matrix_multiplication_cpp_source,
    cuda_sources=fused_group_norm_matrix_multiplication_source,
    functions=["fused_group_norm_matrix_multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.fused_group_norm_matrix_multiplication = fused_group_norm_matrix_multiplication

    def forward(self, A, B):
        return self.fused_group_norm_matrix_multiplication.fused_group_norm_matrix_multiplication_cuda(A, B)