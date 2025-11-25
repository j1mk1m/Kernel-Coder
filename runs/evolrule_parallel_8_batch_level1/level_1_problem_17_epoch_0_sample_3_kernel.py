import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized matrix multiplication (A (M,K) * B (N,K)^T = C (M,N))
# Using shared memory and tiling for better performance
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fast_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int N, int K) {

    // Tile size (block dimensions)
    const int TILE_SIZE = 32;
    __shared__ scalar_t shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t shared_B[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Each thread computes one element of the output block
    // Compute the row and column indexes of the output element
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    scalar_t sum = 0.0;

    // Number of tiles in the K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load data into shared memory
        int a_row = by * TILE_SIZE + ty;
        int a_col = t * TILE_SIZE + tx;
        shared_A[ty][tx] = (a_col < K) ? A[a_row * K + a_col] : 0.0;

        int b_row = t * TILE_SIZE + tx;
        int b_col = bx * TILE_SIZE + tx; // Note: B is transposed, so original B is NxK, here B is stored as (N,K) but accessed as K,N
        shared_B[ty][tx] = (b_row < K && b_col < N) ? B[b_col * K + b_row] : 0.0; // B is stored as NxK, so B_T is KxN. Thus, B_T[i][j] = B[j][i]

        __syncthreads();

        // Compute the product of the two tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor fast_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M, int N, int K) {

    // Ensure dimensions match
    auto stream = at::cuda::getCurrentCUDAStream();

    // Define grid and block dimensions
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);

    // Output tensor
    auto C = torch::empty({M, N}, A.options());

    // Launch kernel
    fast_matmul_kernel<<<blocks, threads, 0, stream>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, N, K);

    return C;
}
"""

matmul_cpp_source = """
#include <torch/extension.h>

at::Tensor fast_matmul_cuda(
    at::Tensor A,
    at::Tensor B,
    int M, int N, int K);
"""

# Compile the CUDA kernel
fast_matmul = load_inline(
    name="fast_matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["fast_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=['-lineinfo', '-O3'],
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.M = 1024 * 2
        self.K = 4096 * 2
        self.N = 2048 * 2
        self.fast_matmul = fast_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.fast_matmul.fast_matmul_cuda(A, B, self.M, self.N, self.K)

def get_inputs():
    # Generate inputs on CUDA
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(N, K, device='cuda')
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed