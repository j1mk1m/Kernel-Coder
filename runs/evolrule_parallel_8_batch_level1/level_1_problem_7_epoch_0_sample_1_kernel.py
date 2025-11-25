import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32  // Tile size for shared memory
#define BLOCK_SIZE 32  // Block size (must be multiple of TILE_DIM)

template <typename scalar_t>
__global__ void matrix_mult_kernel(const scalar_t* __restrict__ A,
                                  const scalar_t* __restrict__ B,
                                  scalar_t* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ scalar_t s_A[TILE_DIM][TILE_DIM];  // Shared memory for A tiles
    __shared__ scalar_t s_B[TILE_DIM][TILE_DIM];  // Shared memory for B tiles

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    scalar_t value = 0;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tiles into shared memory
        int k_start = t * TILE_DIM;
        int k_end = min(k_start + TILE_DIM, K);

        // Load A block into shared memory
        if (row < M && t * TILE_DIM + threadIdx.x < K) {
            s_A[threadIdx.y][threadIdx.x] = 
                A[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B block into shared memory (transposed)
        if (t * TILE_DIM + threadIdx.y < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute the dot product for this tile
        for (int i = 0; i < TILE_DIM; ++i) {
            value += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_DIM, TILE_DIM);  
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, 
               (M + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_ALL_TYPES_AND_HALF(A.scalar_type(), "matrix_mult_cuda", ([&] {
        matrix_mult_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}
"""

matrix_mul_cpp_source = (
    "torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);"
)

matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=["-gencode=arch=compute_86,code=sm_86"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A.cuda(), B.cuda())