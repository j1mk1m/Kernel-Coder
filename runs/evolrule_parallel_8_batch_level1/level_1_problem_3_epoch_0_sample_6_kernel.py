import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the batched matrix multiplication kernel with shared memory and tiling
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TS 32  // Tile size for output matrix (TSxTS)
#define TB 128 // Tile size for inner dimension (k)

__global__ void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n,
    int lda,  // Leading dimension of A (m)
    int ldb,  // Leading dimension of B (k)
    int ldc    // Leading dimension of C (m)
) {
    int batch_idx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.z;

    int row_start = tile_row * TS;
    int col_start = tile_col * TS;
    int row_in_tile = ty;
    int col_in_tile = tx;

    int row = row_start + row_in_tile;
    int col = col_start + col_in_tile;

    if (row >= m || col >= n)
        return;

    __shared__ float shared_A[TS][TB];
    __shared__ float shared_B[TB][TS];

    float C_value = 0.0f;

    for (int p = 0; p < (k + TB - 1) / TB; ++p) {
        int k_start = p * TB;
        int a_col = k_start + tx;
        int a_row = row_start + row_in_tile;

        if (a_col < k) {
            shared_A[row_in_tile][tx] = A[batch_idx * m * k + a_row * k + a_col];
        } else {
            shared_A[row_in_tile][tx] = 0.0f;
        }

        int b_row = k_start + ty;
        int b_col = col_start + col_in_tile;

        if (b_row < k && b_col < n) {
            shared_B[ty][col_in_tile] = B[batch_idx * k * n + b_row * n + b_col];
        } else {
            shared_B[ty][col_in_tile] = 0.0f;
        }

        __syncthreads();

        int k_load = min(TB, k - k_start);
        for (int i = 0; i < k_load; ++i) {
            C_value += shared_A[row_in_tile][i] * shared_B[i][col_in_tile];
        }

        __syncthreads();
    }

    C[batch_idx * m * n + row * n + col] = C_value;
}

torch::Tensor batched_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    const int batch_size = A.size(0);
    const int m = A.size(1);
    const int k = A.size(2);
    const int n = B.size(2);

    TORCH_CHECK(B.size(0) == batch_size);
    TORCH_CHECK(B.size(1) == k);

    auto C = torch::empty({batch_size, m, n}, A.options());

    const int num_tiles_m = (m + TS - 1) / TS;
    const int num_tiles_n = (n + TS - 1) / TS;

    dim3 threadsPerBlock(TS, TS);
    dim3 numBlocks(batch_size, num_tiles_m, num_tiles_n);

    batched_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        k,
        n,
        k,  // lda = k
        n,  // ldb = n
        n    // ldc = n
    );

    cudaDeviceSynchronize();
    return C;
}
"""

batched_matmul_cpp_source = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the CUDA code inline
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)

# Define problem dimensions
batch_size = 128
m = 128 * 4  # 512
k = 256 * 4  # 1024
n = 512 * 4  # 2048

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]

def get_init_inputs():
    return []