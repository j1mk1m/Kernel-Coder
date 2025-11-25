import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication of A (MxK) and B.T (KxN)
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

template <typename T>
__global__ void matmul_transposed_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M,
    int N,
    int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    T sum = 0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "matmul_transposed_cuda", [&] {
        matmul_transposed_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    });

    cudaDeviceSynchronize();
    return C;
}
"""

matrix_mult_cpp_source = (
    "torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
    extra_cflags=["-DUSE_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transposed = matmul_transposed

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_transposed.matmul_transposed_cuda(A, B)

# Update get_inputs to move tensors to CUDA
def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(N, K).cuda()
    return [A, B]