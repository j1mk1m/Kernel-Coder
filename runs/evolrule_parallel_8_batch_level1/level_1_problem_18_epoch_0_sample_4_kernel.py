import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[k * M + m] * B[n * K + k];
    }
    C[m * N + n] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (A.size(0) != B.size(1)) {
        TORCH_CHECK(false, "Incompatible matrix dimensions for multiplication.");
    }
    int M = A.size(1);
    int K = A.size(0);
    int N = B.size(0);
    auto C = torch::zeros({M, N}, A.options());
    int threads_per_block = 128;
    int num_elements = M * N;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<num_blocks, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda()
        B = B.cuda()
        return matmul_cuda(A, B)