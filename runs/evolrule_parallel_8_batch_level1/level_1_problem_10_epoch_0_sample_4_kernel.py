import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 32

__global__ void matmul3d_kernel(
    const float* A,
    const float* B,
    float* C,
    int N, int M, int K, int L
) {
    int batch = blockIdx.x;
    int c_row = blockIdx.y * BLOCK_SIZE;
    int c_col = blockIdx.z * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = c_row + ty;
    int col = c_col + tx;

    if (row >= M || col >= L) return;

    __shared__ float As[BLOCK_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][BLOCK_SIZE];

    float Cval = 0.0f;

    for (int k_chunk = 0; k_chunk < (K + TILE_SIZE - 1) / TILE_SIZE; k_chunk++) {
        int k_start = k_chunk * TILE_SIZE;

        if (tx < TILE_SIZE) {
            int a_row = c_row + ty;
            int a_col = k_start + tx;
            if (a_row < M && a_col < K) {
                int a_offset = batch * M * K + a_row * K + a_col;
                As[ty][tx] = A[a_offset];
            } else {
                As[ty][tx] = 0.0f;
            }
        }

        if (ty < TILE_SIZE) {
            int b_row = k_start + ty;
            int b_col = c_col + tx;
            if (b_row < K && b_col < L) {
                int b_offset = b_row * L + b_col;
                Bs[ty][tx] = B[b_offset];
            } else {
                Bs[ty][tx] = 0.0f;
            }
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            Cval += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    int c_offset = batch * M * L + row * L + col;
    C[c_offset] = Cval;
}

torch::Tensor matmul3d_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    auto C = torch::empty({N, M, L}, A.options());

    int num_M_tiles = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_L_tiles = (L + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(N, num_M_tiles, num_L_tiles);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matmul3d_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, K, L
    );

    return C;
}
"""

matmul3d_cpp_source = "torch::Tensor matmul3d_cuda(torch::Tensor A, torch::Tensor B);"

matmul3d = load_inline(
    name="matmul3d",
    cuda_sources=matmul3d_source,
    cpp_sources=matmul3d_cpp_source,
    functions=["matmul3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul3d = matmul3d

    def forward(self, A, B):
        return self.matmul3d.matmul3d_cuda(A.contiguous(), B.contiguous())