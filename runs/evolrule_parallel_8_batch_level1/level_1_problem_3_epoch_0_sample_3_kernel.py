import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batch_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void batch_matmul_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int batch_size, int m, int k, int n) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int batch = blockIdx.x;
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.z;

    int i_start = tile_m * TILE_DIM;
    int j_start = tile_n * TILE_DIM;

    __shared__ float shared_A[TILE_DIM][TILE_DIM];
    __shared__ float shared_B[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (int tile_k = 0; tile_k < (k + TILE_DIM - 1) / TILE_DIM; ++tile_k) {
        // Load A tile into shared memory
        int a_row = i_start + ty;
        int a_col = tile_k * TILE_DIM + tx;
        if (a_row < m && a_col < k) {
            shared_A[ty][tx] = A[batch * m * k + a_row * k + a_col];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        int b_row = tile_k * TILE_DIM + ty;
        int b_col = j_start + tx;
        if (b_row < k && b_col < n) {
            shared_B[ty][tx] = B[batch * k * n + b_row * n + b_col];
        } else {
            shared_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the partial sum
        for (int kk = 0; kk < TILE_DIM; ++kk) {
            sum += shared_A[ty][kk] * shared_B[kk][tx];
        }

        __syncthreads();
    }

    int i = i_start + ty;
    int j = j_start + tx;
    if (i < m && j < n) {
        C[batch * m * n + i * n + j] = sum;
    }
}

torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k_A = A.size(2);
    int k_B = B.size(1);
    int n = B.size(2);

    assert(k_A == k_B);
    assert(A.is_contiguous() && B.is_contiguous());
    assert(A.dtype() == B.dtype());

    auto C = torch::empty({batch_size, m, n}, A.options());

    dim3 threads(TILE_DIM, TILE_DIM);
    int tiles_m = (m + TILE_DIM - 1) / TILE_DIM;
    int tiles_n = (n + TILE_DIM - 1) / TILE_DIM;
    dim3 blocks(batch_size, tiles_m, tiles_n);

    batch_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, m, k_A, n);

    cudaDeviceSynchronize();
    return C;
}
"""

batch_matmul_header = """
torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for batched matrix multiplication
batch_matmul = load_inline(
    name="batch_matmul",
    cuda_sources=batch_matmul_source,
    cpp_sources=batch_matmul_header,
    functions=["batch_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_matmul = batch_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batch_matmul.batch_matmul_cuda(A, B)

# Define the problem dimensions
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed