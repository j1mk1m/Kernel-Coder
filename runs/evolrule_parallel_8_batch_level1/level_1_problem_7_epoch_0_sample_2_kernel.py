import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 16384 * 2
N = 16384 * 2
K = 32 * 2

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    int block_row_start = block_row * TILE_WIDTH;
    int block_col_start = block_col * TILE_WIDTH;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row = block_row_start + local_row;
    int col = block_col_start + local_col;

    float sum = 0.0f;

    __shared__ float shared_A[TILE_WIDTH * TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH * TILE_WIDTH];

    for (int k_start = 0; k_start < K; k_start += TILE_WIDTH) {
        // Load A and B tiles into shared memory
        if (block_row_start + local_row < M && k_start + local_col < K) {
            shared_A[local_row * TILE_WIDTH + local_col] = A[(block_row_start + local_row) * K + (k_start + local_col)];
        } else {
            shared_A[local_row * TILE_WIDTH + local_col] = 0.0f;
        }

        if (k_start + local_row < K && block_col_start + local_col < N) {
            shared_B[local_row * TILE_WIDTH + local_col] = B[(k_start + local_row) * N + (block_col_start + local_col)];
        } else {
            shared_B[local_row * TILE_WIDTH + local_col] = 0.0f;
        }

        __syncthreads();

        // Compute the partial sum for this chunk
        for (int k_in_chunk = 0; k_in_chunk < TILE_WIDTH; ++k_in_chunk) {
            int k = k_start + k_in_chunk;
            if (k < K) {
                sum += shared_A[local_row * TILE_WIDTH + k_in_chunk] * shared_B[k_in_chunk * TILE_WIDTH + local_col];
            }
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K_A = A.size(1);
    const int K_B = B.size(0);
    const int N = B.size(1);

    if (K_A != K_B) {
        TORCH_CHECK(false, "Incompatible matrix dimensions");
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor C = torch::empty({M, N}, options);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    int block_rows = (M + TILE_WIDTH - 1) / TILE_WIDTH;
    int block_cols = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 blocks(block_rows, block_cols);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K_A);

    return C;
}
"""

matmul_cuda_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cuda_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)