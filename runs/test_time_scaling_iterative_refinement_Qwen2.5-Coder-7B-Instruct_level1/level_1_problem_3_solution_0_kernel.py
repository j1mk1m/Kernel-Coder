import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batch_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_matmul_kernel(const float* A, const float* B, float* C, int batch_size, int m, int k, int n) {
    int batch_idx = blockIdx.z;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && row_idx < m && col_idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[batch_idx * m * k + row_idx * k + i] * B[batch_idx * k * n + i * n + col_idx];
        }
        C[batch_idx * m * n + row_idx * n + col_idx] = sum;
    }
}

__global__ void batch_matmul_shared_kernel(const float* A, const float* B, float* C, int batch_size, int m, int k, int n) {
    extern __shared__ float s_A[];
    extern __shared__ float s_B[];

    int batch_idx = blockIdx.z;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;
    float pval = 0.0f;

    // Load data into shared memory
    if (row_idx < m && tile_col < k) {
        s_A[tile_row * blockDim.x + tile_col] = A[batch_idx * m * k + row_idx * k + tile_col];
    } else {
        s_A[tile_row * blockDim.x + tile_col] = 0.0f;
    }

    if (tile_row < k && col_idx < n) {
        s_B[tile_row * blockDim.x + tile_col] = B[batch_idx * k * n + tile_row * n + col_idx];
    } else {
        s_B[tile_row * blockDim.x + tile_col] = 0.0f;
    }

    __syncthreads();

    // Perform reduction within the tile
    for (int i = 0; i < blockDim.x; ++i) {
        pval += s_A[tile_row * blockDim.x + i] * s_B[i * blockDim.x + tile_col];
    }

    __syncthreads();

    // Write result back to global memory
    if (row_idx < m && col_idx < n) {
        atomicAdd(&C[batch_idx * m * n + row_idx * n + col_idx], pval);
    }
}

torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);
    auto C = torch::zeros({batch_size, m, n}, A.options());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x,
                         (m + threads_per_block.y - 1) / threads_per_block.y,
                         batch_size);

    batch_matmul_kernel<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, m, k, n);

    return C;
}

torch::Tensor batch_matmul_shared_cuda(torch::Tensor A, torch::Tensor B) {
    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);
    auto C = torch::zeros({batch_size, m, n}, A.options());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x,
                         (m + threads_per_block.y - 1) / threads_per_block.y,
                         batch_size);

    int shared_mem_size = 2 * sizeof(float) * threads_per_block.x * threads_per_block.y;
    batch_matmul_shared_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, m, k, n);

    return C;
}
"""

batch_matmul_cpp_source = (
    "torch::Tensor batch_matmul_cuda(torch::Tensor A, torch::Tensor B);"
    "torch::Tensor batch_matmul_shared_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for batched matrix multiplication
batch_matmul = load_inline(
    name="batch_matmul",
    cpp_sources=batch_matmul_cpp_source,
    cuda_sources=batch_matmul_source,
    functions=["batch_matmul_cuda", "batch_matmul_shared_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batch_matmul = batch_matmul
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batch_matmul.batch_matmul_shared_cuda(A, B)