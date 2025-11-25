import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for triangular matrix multiplication
triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void triangular_matmul_kernel(const scalar_t* __restrict__ A,
                                        const scalar_t* __restrict__ B,
                                        scalar_t* __restrict__ C,
                                        int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    scalar_t sum = 0;
    for (int k = 0; k <= min(row, col); ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    if (row >= col) {
        C[row * N + col] = sum;
    }
}

torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "triangular_matmul_cuda", ([&] {
        triangular_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), N);
    }));

    return C;
}
"""

triangular_matmul_cpp_source = (
    "torch::Tensor triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the CUDA kernel
triangular_matmul = load_inline(
    name="triangular_matmul",
    cpp_sources=triangular_matmul_cpp_source,
    cuda_sources=triangular_matmul_source,
    functions=["triangular_matmul_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.triangular_matmul = triangular_matmul

    def forward(self, A, B):
        return self.triangular_matmul.triangular_matmul_cuda(A, B)

def get_inputs():
    M = 4096
    A = torch.rand(M, M).tril_()
    B = torch.rand(M, M).tril_()
    return [A.cuda(), B.cuda()]

def get_init_inputs():
    return []