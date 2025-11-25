import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <cuda_runtime.h>

__global__ void batched_matmul(
    const float* A, const float* B, float* C,
    int batch_size, int m, int k, int n) {

    int batch = blockIdx.x;
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_size = 32;

    int row_start = tile_row * tile_size;
    int col_start = tile_col * tile_size;

    __shared__ float shared_A[tile_size][tile_size];
    __shared__ float shared_B[tile_size][tile_size];

    float sum = 0.0;

    for (int p = 0; p < (k + tile_size - 1) / tile_size; p++) {

        // Load A tile into shared memory
        int a_row = row_start + ty;
        int a_col = p * tile_size + tx;
        if (a_row < m && a_col < k) {
            shared_A[ty][tx] = A[batch * m * k + a_row * k + a_col];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        int b_row = p * tile_size + tx;
        int b_col = col_start + ty;
        if (b_row < k && b_col < n) {
            shared_B[tx][ty] = B[batch * k * n + b_row * n + b_col];
        } else {
            shared_B[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute the product for this tile
        for (int i = 0; i < tile_size; i++) {
            sum += shared_A[ty][i] * shared_B[i][tx];
        }

        __syncthreads();
    }

    // Write the result to global memory
    int row = row_start + ty;
    int col = col_start + tx;
    if (row < m && col < n) {
        C[batch * m * n + row * n + col] = sum;
    }
}
"""

# The corresponding C++ header declaration
batched_matmul_cpp = """
extern "C" __global__ void batched_matmul(
    const float*, const float*, float*, int, int, int, int);
"""

# Compile the CUDA code
batched_matmul_cuda = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        batch_size_val, m_val, k_val = A.shape
        _, k_B_val, n_val = B.shape
        assert batch_size_val == B.shape[0], "Batch sizes must match"
        assert k_val == k_B_val, "Inner dimensions must match"
        # Ensure contiguous memory for efficient access
        A = A.contiguous()
        B = B.contiguous()
        # Allocate output tensor
        C = torch.empty(batch_size_val, m_val, n_val, device=A.device, dtype=A.dtype)
        # Define block and grid dimensions
        tile_size = 32
        threads_per_block = (tile_size, tile_size)
        num_tiles_m = (m_val + tile_size - 1) // tile_size
        num_tiles_n = (n_val + tile_size - 1) // tile_size
        blocks_per_grid = (
            batch_size_val,  # blockIdx.x is batch index
            num_tiles_m,     # blockIdx.y is tile row index
            num_tiles_n      # blockIdx.z is tile column index
        )
        # Launch the kernel
        batched_matmul_cuda.batched_matmul[
            blocks_per_grid, threads_per_block
        ](A.data_ptr(), B.data_ptr(), C.data_ptr(), batch_size_val, m_val, k_val, n_val)
        return C

def get_inputs():
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed