import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

upper_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void upper_triangular_matmul(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * N) return;

    int i = tid / N;
    int j = tid % N;

    if (i > j) return;

    float sum = 0.0f;
    for (int k = i; k <= j; ++k) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    int num_threads = N * N;
    int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;

    upper_triangular_matmul<<<num_blocks, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

upper_triangular_matmul_cpp_src = """
torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

upper_triangular_matmul = load_inline(
    name="upper_triangular_matmul",
    cpp_sources=upper_triangular_matmul_cpp_src,
    cuda_sources=upper_triangular_matmul_source,
    functions=["upper_triangular_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.upper_triangular_matmul = upper_triangular_matmul

    def forward(self, A, B):
        return self.upper_triangular_matmul.upper_triangular_matmul_cuda(A, B)

def get_inputs():
    N = 4096
    A = torch.triu(torch.rand(N, N).cuda())
    B = torch.triu(torch.rand(N, N).cuda())
    return [A, B]

def get_init_inputs():
    return []