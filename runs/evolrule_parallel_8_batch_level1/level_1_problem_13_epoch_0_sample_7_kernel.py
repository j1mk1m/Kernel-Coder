from torch.utils.cpp_extension import load_inline
import torch
import torch.nn as nn

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_sym_cuda_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float C_value = 0.0f;

    for (int k_block = 0; k_block < (N / TILE_WIDTH); ++k_block) {
        __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
        __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

        // Load A tile
        int A_row = row;
        int A_col = k_block * TILE_WIDTH + threadIdx.x;
        if (A_row < N && A_col < N) {
            shared_A[threadIdx.y][threadIdx.x] = A[A_row * N + A_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        int B_row = k_block * TILE_WIDTH + threadIdx.y;
        int B_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (B_row < N && B_col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[B_row * N + B_col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial products
        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = C_value;
    }
}

torch::Tensor matmul_sym_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    assert(A.sizes() == torch::IntArrayRef({N, N}));
    assert(B.sizes() == torch::IntArrayRef({N, N}));
    assert(A.is_contiguous() && B.is_contiguous() && A.device() == B.device());

    auto C = torch::empty({N, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_sym_cuda_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_sym_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_sym_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A, B):
        return self.matmul_cuda.matmul_sym_cuda(A, B)