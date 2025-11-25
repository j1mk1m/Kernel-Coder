import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define N 32
#define T 32

__global__ void tiled_matmul(float* A, float* B, float* C, int M) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int block_row_start = block_row * T;
    int block_col_start = block_col * T;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float shared_A[T][N];
    __shared__ float shared_B[N][T];

    // Load A into shared memory
    int a_global_row = block_row_start + ty;
    int a_global_col = tx;
    if (a_global_row < M && a_global_col < N) {
        shared_A[ty][tx] = A[a_global_row * N + a_global_col];
    }

    // Load B into shared memory
    int b_global_row = ty;
    int b_global_col = block_col_start + tx;
    if (b_global_row < N && b_global_col < M) {
        shared_B[ty][tx] = B[b_global_row * M + b_global_col];
    }

    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += shared_A[ty][k] * shared_B[k][tx];
    }

    int c_row = block_row_start + ty;
    int c_col = block_col_start + tx;
    if (c_row < M && c_col < M) {
        C[c_row * M + c_col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int N = A.size(1);
    assert(N == 32); // Ensure correct dimensions
    auto C = torch::empty({M, M}, A.options());

    dim3 threads(T, T);
    int blocks_per_dim = (M + T - 1) / T;
    dim3 blocks(blocks_per_dim, blocks_per_dim);

    tiled_matmul<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A, B)

def get_inputs():
    M = 16384 * 2
    N = 16 * 2
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []