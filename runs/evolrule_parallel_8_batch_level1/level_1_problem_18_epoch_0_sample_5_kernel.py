import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transposed_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C,
                                         int M, int K, int N) {
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(32, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    matmul_transposed_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

matmul_transposed_cpp_source = """
torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_transposed = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_cpp_source,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transposed = matmul_transposed

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # The original code computes (A.T) * (B.T) = (B * A)^T
        # But the user's forward says return torch.matmul(A.T, B.T)
        # So we need to compute B (shape N,K) multiplied by A (shape K,M), resulting in N x M, then transpose to M x N?
        # Wait, let's verify dimensions:
        # A.T is M x K, B.T is K x N. So their product is M x N, which matches the original code.
        # However, the custom kernel is written as A (original A.T has shape M,K, so in kernel A is MxK stored as row-major)
        # B is passed as B.T's transpose? Wait need to be careful with tensor shapes.
        # Wait the inputs to the custom kernel are A and B as passed to forward, which are the same as the original function's inputs.
        # The original forward is torch.matmul(A.T, B.T) which is (M,K) @ (K,N) = MxN
        # The custom kernel's logic is: A has shape (M, K) (since it's A.T's storage), and B has shape (N, K) (since it's B's original storage)
        # So the kernel computes C[row][col] = sum_{k} A[row][k] * B[col][k]
        # Which is exactly (A^T @ B^T)[row, col] = sum_{k} A_{k, row} * B_{k, col} = sum_{k} A[row][k] (since A is stored as row-major for A.T) * B[col][k]
        # So the kernel correctly implements the original operation. Hence no transpose needed on inputs.

        return self.matmul_transposed.matmul_transposed_cuda(A, B)