import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for lower triangular matrix multiplication
ltmatmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void lower_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one element of the lower triangular part
    if (idx < N * N) {
        int i = idx / N;
        int j = idx % N;
        if (i < j) return; // Only compute lower triangular elements (i >= j)

        float sum = 0.0;
        // Sum over k from j to i (inclusive)
        for (int k = j; k <= i; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[idx] = sum;
    }
}

torch::Tensor lower_triangular_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    const int N = A.size(0);
    const int total_elements = N * N;

    // Output tensor
    auto C = torch::zeros({N, N}, A.options());

    // Number of blocks needed
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    lower_triangular_matmul_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

ltmatmul_cpp_source = "torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the CUDA kernel
ltmatmul = load_inline(
    name="ltmatmul",
    cpp_sources=ltmatmul_cpp_source,
    cuda_sources=ltmatmul_source,
    functions=["lower_triangular_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.ltmatmul = ltmatmul

    def forward(self, A, B):
        return self.ltmatmul.lower_triangular_matmul_cuda(A, B)

def get_inputs():
    M = 4096
    A = torch.rand(M, M).tril_()
    B = torch.rand(M, M).tril_()
    return [A.cuda(), B.cuda()]

def get_init_inputs():
    return []