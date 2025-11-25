import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 256
N = 256
K = 131072 * 4

# Define the custom CUDA kernel for large K matrix multiplication
matmul_large_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_large_K_cuda(const float* a, const float* b, float* c, int M, int N, int K) {
    __shared__ float A_sh[TILE_SIZE][TILE_SIZE];
    __shared__ float B_sh[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.x * TILE_SIZE;
    int block_col = blockIdx.y * TILE_SIZE;

    int row = block_row + ty;
    int col = block_col + tx;

    float partial_sum = 0.0f;

    for (int chunk = 0; chunk < K; chunk += TILE_SIZE) {
        int col_A = chunk + tx;
        int row_A = block_row + ty;

        int row_B = chunk + tx;
        int col_B = block_col + ty;

        if (col_A < K && row_A < M) {
            A_sh[ty][tx] = a[row_A * K + col_A];
        } else {
            A_sh[ty][tx] = 0.0f;
        }

        if (row_B < K && col_B < N) {
            B_sh[tx][ty] = b[row_B * N + col_B];
        } else {
            B_sh[tx][ty] = 0.0f;
        }

        __syncthreads();

        for (int l = 0; l < TILE_SIZE; ++l) {
            partial_sum += A_sh[ty][l] * B_sh[l][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = partial_sum;
    }
}

torch::Tensor matmul_large_K_cuda(torch::Tensor a, torch::Tensor b) {
    int M = a.size(0);
    int K_a = a.size(1);
    int K_b = b.size(0);
    int N = b.size(1);
    assert(K_a == K_b && "Matrix dimensions must match.");

    auto c = torch::empty({M, N}, a.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (M + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_large_K_cuda<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K_a
    );

    return c;
}
"""

matmul_large_cpp_source = (
    "torch::Tensor matmul_large_K_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code
matmul_large = load_inline(
    name="matmul_large",
    cpp_sources=matmul_large_cpp_source,
    cuda_sources=matmul_large_source,
    functions=["matmul_large_K_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_large = matmul_large

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda()
        B = B.cuda()
        return self.matmul_large.matmul_large_K_cuda(A, B)