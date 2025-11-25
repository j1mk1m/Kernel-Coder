import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float C_value = 0.0f;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        int a_col = m * TILE_WIDTH + tx;
        int a_row = row;
        if (a_col < K && a_row < M) {
            As[ty][tx] = A[a_row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        int b_row = m * TILE_WIDTH + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            Bs[ty][tx] = B[b_row * N + b_col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}

torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M_val = A.size(0);
    int K_val = A.size(1);
    int N_val = B.size(1);

    auto C = torch::empty({M_val, N_val}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N_val + TILE_WIDTH - 1) / TILE_WIDTH,
        (M_val + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M_val, K_val, N_val);

    return C;
}
"""

matmul_cuda_h = """
torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=matmul_cuda_h,
    cuda_sources=matmul_cuda_source,
    functions=["custom_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_matmul = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_matmul.custom_matmul_cuda(A.t(), B)

def get_inputs():
    A = torch.rand(K, M).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []