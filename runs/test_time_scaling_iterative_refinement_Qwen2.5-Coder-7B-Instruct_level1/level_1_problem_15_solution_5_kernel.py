import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication of lower triangular matrices
lower_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lower_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k <= min(row, col); ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const int block_size = 16;
    const int grid_x = (N + block_size - 1) / block_size;
    const int grid_y = (N + block_size - 1) / block_size;

    lower_triangular_matmul_kernel<<<grid_x, grid_y, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

lower_triangular_matmul_cpp_source = (
    "torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication of lower triangular matrices
lower_triangular_matmul = load_inline(
    name="lower_triangular_matmul",
    cpp_sources=lower_triangular_matmul_cpp_source,
    cuda_sources=lower_triangular_matmul_source,
    functions=["lower_triangular_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return lower_triangular_matmul.lower_triangular_matmul_cuda(A, B)

# Test the optimized architecture
M = 4096
A = torch.rand(M, M).cuda()
B = torch.rand(M, M).cuda()
A = torch.tril(A).cuda()
B = torch.tril(B).cuda()

model = Model().cuda()
output_ref = model(A, B)

model_new = ModelNew().cuda()
output_optimized = model_new(A, B)

print("Reference output:", output_ref.sum())
print("Optimized output:", output_optimized.sum())

assert torch.allclose(output_ref, output_optimized), "The outputs do not match."