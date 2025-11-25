import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32

__global__ void batched_matmul_kernel(
    const float* A, const float* B, float* C,
    int batch_size, int m, int k, int n) {

    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = 0.0;
    for (int e = 0; e < k; e++) {
        sum += A[batch * m * k + row * k + e] *
               B[batch * k * n + e * n + col];
    }
    C[batch * m * n + row * n + col] = sum;
}

torch::Tensor batched_matmul_cuda(
    torch::Tensor A, torch::Tensor B) {

    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);

    auto C = torch::empty({batch_size, m, n}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size);

    batched_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, m, k, n);

    return C;
}
"""

batched_matmul_cpp_source = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)

def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []