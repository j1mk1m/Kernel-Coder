import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

lower_tril_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void lower_tril_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0;
        return;
    }

    scalar_t sum = 0.0;
    for (int k = col; k <= row; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

torch::Tensor lower_tril_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const int threads = 32;
    dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "lower_tril_matmul_cuda", ([&]{
        lower_tril_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N
        );
    }));

    return C;
}
"""

lower_tril_matmul_cpp_source = """
torch::Tensor lower_tril_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for lower triangular matrix multiplication
lower_tril_matmul = load_inline(
    name="lower_tril_matmul",
    cpp_sources=lower_tril_matmul_cpp_source,
    cuda_sources=lower_tril_matmul_source,
    functions=["lower_tril_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lower_tril_matmul = lower_tril_matmul

    def forward(self, A, B):
        return self.lower_tril_matmul.lower_tril_matmul_cuda(A, B)