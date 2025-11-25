import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for upper triangular matrix multiplication
upper_tri_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void upper_tri_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int N) {
    // Each thread computes C[i][j] where i <= j
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    
    // Map idx to i and j such that i <= j
    int i = idx / N;
    int j = idx % N;
    if (i > j) return; // Skip lower triangle
    
    scalar_t sum = 0;
    // Sum from k = i to k = j
    for (int k = i; k <= j; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

torch::Tensor upper_tri_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int numel = N * N;
    auto C = torch::zeros({N, N}, A.options());
    
    const int threads_per_block = 256;
    const int blocks_per_grid = (numel + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "upper_tri_matmul_cuda", ([&] {
        upper_tri_matmul_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N);
    }));
    
    return C;
}
"""

upper_tri_matmul_cpp_source = (
    "torch::Tensor upper_tri_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
upper_tri_matmul = load_inline(
    name="upper_tri_matmul",
    cpp_sources=upper_tri_matmul_cpp_source,
    cuda_sources=upper_tri_matmul_source,
    functions=["upper_tri_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.upper_tri_matmul = upper_tri_matmul

    def forward(self, A, B):
        return self.upper_tri_matmul.upper_tri_matmul_cuda(A, B)