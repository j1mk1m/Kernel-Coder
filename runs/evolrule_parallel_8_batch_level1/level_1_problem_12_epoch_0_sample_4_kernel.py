import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diag(A) @ B operation
diag_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_mult_kernel(const float* A, const float* B, float* C, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        int i = idx / M;
        int j = idx % M;
        C[idx] = A[i] * B[idx];
    }
}

torch::Tensor diag_mult_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = B.size(1);
    auto C = torch::empty({N, M}, A.options());
    const int threads_per_block = 256;
    const int blocks_per_grid = (N * M + threads_per_block - 1) / threads_per_block;
    diag_mult_kernel<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
    return C;
}
"""

diag_mult_cpp_source = "torch::Tensor diag_mult_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
diag_mult = load_inline(
    name="diag_mult",
    cpp_sources=diag_mult_cpp_source,
    cuda_sources=diag_mult_source,
    functions=["diag_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag_mult = diag_mult

    def forward(self, A, B):
        return self.diag_mult.diag_mult_cuda(A, B)

M = 4096
N = 4096

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []