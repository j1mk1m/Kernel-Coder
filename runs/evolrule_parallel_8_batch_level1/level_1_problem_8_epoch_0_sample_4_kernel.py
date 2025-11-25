import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 8205
K = 2949
N = 5921

# Define the CUDA kernel for optimized matrix multiplication
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrix_mult_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float C_value = 0.0;

    for (int k_tile = 0; k_tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++k_tile) {
        __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
        __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

        int a_row = row;
        int a_col = k_tile * TILE_WIDTH + threadIdx.x;
        int b_row = k_tile * TILE_WIDTH + threadIdx.y;
        int b_col = col;

        // Load tiles into shared memory
        if (a_col < K && threadIdx.y < TILE_WIDTH) {
            shared_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (b_row < K && b_col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Compute partial sum
        for (int t = 0; t < TILE_WIDTH; ++t) {
            C_value += shared_A[threadIdx.y][t] * shared_B[t][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}

torch::Tensor matrix_mult_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M,
    int K,
    int N
) {
    auto stream = A.cuda_stream();

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matrix_mult_kernel<<<blocks, threads, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

matrix_mult_cpp_source = """
torch::Tensor matrix_mult_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M,
    int K,
    int N
);
"""

# Compile the CUDA code
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A, B, M, K, N)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []