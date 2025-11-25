import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

upper_triangular_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void upper_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (N * (N + 1)) / 2) return;

    // Compute i and j from idx
    int i = (int)( (sqrt(8.0f * idx + 1) - 1) / 2 );
    int T_prev = (i-1)*i/2;
    int offset = idx - T_prev;
    int j = i + offset;

    float sum = 0.0f;
    for (int k = i; k <= j; k++) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    torch::Tensor C = torch::zeros({N, N}, A.options());

    int threads_per_block = 256;
    int num_elements = (N * (N + 1)) / 2;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    upper_triangular_matmul_kernel<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();

    return C;
}
"""

upper_triangular_matmul_cpp_source = (
    "torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

upper_triangular_matmul = load_inline(
    name="upper_triangular_matmul",
    cpp_sources=upper_triangular_matmul_cpp_source,
    cuda_sources=upper_triangular_matmul_source,
    functions=["upper_triangular_matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.upper_triangular_matmul = upper_triangular_matmul

    def forward(self, A, B):
        return self.upper_triangular_matmul.upper_triangular_matmul_cuda(A, B)