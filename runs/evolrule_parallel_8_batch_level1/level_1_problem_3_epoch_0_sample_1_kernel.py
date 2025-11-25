import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void batched_matmul_kernel(
    const float* A, const float* B, float* C,
    int batch_size, int m, int k, int n) {
    int batch = blockIdx.x;
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = tile_row * TILE_DIM + ty;
    int col = tile_col * TILE_DIM + tx;

    float Cvalue = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int p = 0; p < (k + TILE_DIM - 1) / TILE_DIM; p++) {
        // Load A's tile into shared memory
        int aRow = row;
        int aCol = p * TILE_DIM + tx;
        if (aCol < k && ty < TILE_DIM) {
            As[ty][tx] = A[batch * m * k + aRow * k + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B's tile into shared memory
        int bRow = p * TILE_DIM + ty;
        int bCol = col;
        if (bRow < k && tx < TILE_DIM) {
            Bs[ty][tx] = B[batch * k * n + bRow * n + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int i = 0; i < TILE_DIM; i++) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }

        // Removed redundant __syncthreads()
    }

    if (row < m && col < n) {
        C[batch * m * n + row * n + col] = Cvalue;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check input dimensions
    if (A.dim() != 3 || B.dim() != 3) {
        torch::TORCH_CHECK(false, "Input tensors must be 3-dimensional");
    }
    int batch_size = A.size(0);
    int m = A.size(1);
    int k_A = A.size(2);
    int k_B = B.size(1);
    int n = B.size(2);
    torch::TORCH_CHECK(k_A == k_B, "Incompatible matrix dimensions for batched multiplication.");

    auto C = torch::empty({batch_size, m, n}, A.options());

    // Calculate grid and block dimensions
    int num_tiles_m = (m + TILE_DIM - 1) / TILE_DIM;
    int num_tiles_n = (n + TILE_DIM - 1) / TILE_DIM;
    dim3 blocks(batch_size, num_tiles_m, num_tiles_n);
    dim3 threads(TILE_DIM, TILE_DIM);

    // Launch the kernel
    batched_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, m, k_A, n
    );

    return C;
}
"""

batched_matmul_cpp_source = R"""extern "C" {
    torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);
}
"""

# Compile the CUDA code
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)