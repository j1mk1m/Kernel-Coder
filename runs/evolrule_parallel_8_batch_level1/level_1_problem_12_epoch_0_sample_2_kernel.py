import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        int i = idx / M;
        int j = idx % M;
        C[idx] = A[i] * B[idx];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = B.size(1);
    assert(A.sizes()[0] == N, "A must be 1D of size N");
    assert(B.sizes()[0] == N && B.sizes()[1] == M, "B must be NxM");
    auto output = torch::empty_like(B);
    int block_size = 256;
    int num_elements = N * M;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    diag_matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), N, M);
    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
extern "C" {
    torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);
}
"""

diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["diag_matmul_cuda"],
    verbose=False,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__"],
    extra_cuda_cflags=["-D__CUDA_NO_HALF_OPERATORS__"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        A = A.cuda()
        B = B.cuda()
        return self.diag_matmul.diag_matmul_cuda(A, B)