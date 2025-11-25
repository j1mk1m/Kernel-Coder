import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_mult_source = """
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
    auto C = torch::empty({N, M}, B.options());

    const int block_size = 256;
    const int total_elements = N * M;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    diag_matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

    return C;
}
"""

diag_mult_cpp_source = (
    "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

diag_mult = load_inline(
    name="diag_mult",
    cpp_sources=diag_mult_cpp_source,
    cuda_sources=diag_mult_source,
    functions=["diag_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_mult = diag_mult

    def forward(self, A, B):
        return self.diag_mult.diag_matmul_cuda(A, B)