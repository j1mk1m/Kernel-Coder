import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_kernel(float* C, const float* A, const float* B, int K, int M, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        int a_col = m * TILE_WIDTH + tx;
        int a_row = Row;

        if (a_col < K && a_row < M) {
            ds_A[ty][tx] = A[a_col * M + a_row];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        int b_row = a_col;
        int b_col = Col;

        if (b_row < K && b_col < N) {
            ds_B[ty][tx] = B[b_row * N + b_col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (Row < M && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matmul_kernel<<<blocks, threads>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), K, M, N);

    return C;
}
"""

matmul_header = """
torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_header,
    cuda_sources=matmul_source,
    functions=["custom_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_matmul = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return self.custom_matmul.custom_matmul_cuda(A, B)