import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TB 32
#define N 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M) {
    __shared__ float shared_A[32][32];  // TB rows of A (each row has N elements)
    __shared__ float shared_B[32][32];  // TB columns of B (each column has N elements)

    // Calculate the tile's position
    int i0 = blockIdx.x * TB;
    int j0 = blockIdx.y * TB;

    // Load A into shared memory
    {
        int row_in_tile = threadIdx.x / N;
        int k = threadIdx.x % N;

        int row_A = i0 + row_in_tile;
        if (row_A < M) {
            shared_A[row_in_tile][k] = A[row_A * N + k];
        } else {
            shared_A[row_in_tile][k] = 0.0f;
        }
    }

    // Load B into shared memory
    {
        int row_in_B = threadIdx.x % N;
        int col_in_tile = threadIdx.x / N;

        int col_B = j0 + col_in_tile;
        if (col_B < M) {
            shared_B[col_in_tile][row_in_B] = B[row_in_B * M + col_B];
        } else {
            shared_B[col_in_tile][row_in_B] = 0.0f;
        }
    }

    __syncthreads();

    // Compute the element
    int i_in_tile = threadIdx.x % TB;
    int j_in_tile = threadIdx.x / TB;

    int i = i0 + i_in_tile;
    int j = j0 + j_in_tile;

    if (i < M && j < M) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += shared_A[i_in_tile][k] * shared_B[j_in_tile][k];
        }
        C[i * M + j] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M_A = A.size(0);
    const int N_A = A.size(1);
    const int M_B = B.size(0);
    const int N_B = B.size(1);

    // Check dimensions: A is M x N, B is N x M
    assert(N_A == M_B && "Matrix dimensions must agree for multiplication");
    assert(N_A == N && "N should be 32");
    assert(M_A == M_B && "The inner dimensions must match");

    int M = M_A;

    // Output tensor
    auto C = torch::zeros({M, M}, A.options());

    // Launch configuration
    dim3 threads(TB * TB);
    int blocks_per_side = (M + TB - 1) / TB;
    dim3 blocks(blocks_per_side, blocks_per_side);

    // Launch kernel
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul

    def forward(self, A, B):
        return self.matmul_cuda.matmul_cuda(A, B)