import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TS 16
#define TILE_K 32

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float sA[TS][TILE_K];
    __shared__ float sB[TILE_K][TS];

    int block_row = blockIdx.y * TS;
    int block_col = blockIdx.x * TS;
    int row = block_row + threadIdx.y;
    int col = block_col + threadIdx.x;

    if (row >= M || col >= N) return;

    float Cval = 0.0f;

    for (int chunk = 0; chunk < (K + TILE_K - 1) / TILE_K; chunk++) {
        int k_start = chunk * TILE_K;
        int k_end = min(k_start + TILE_K, K);

        // Load A tile into shared memory
        for (int i = 0; i < (TILE_K / blockDim.x); i++) {
            int col_A = k_start + threadIdx.x + i * blockDim.x;
            if (col_A < k_end) {
                sA[threadIdx.y][threadIdx.x + i * blockDim.x] = A[(block_row + threadIdx.y) * K + col_A];
            }
        }

        // Load B tile into shared memory
        for (int i = 0; i < (TILE_K / blockDim.y); i++) {
            int row_B = k_start + threadIdx.y + i * blockDim.y;
            if (row_B < k_end) {
                sB[threadIdx.y + i * blockDim.y][threadIdx.x] = B[row_B * N + (block_col + threadIdx.x)];
            }
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < (k_end - k_start); k++) {
            Cval += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cval;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K) {
    auto output = torch::empty({M, N}, A.options());

    dim3 threads(TS, TS);
    dim3 blocks((N + TS - 1) / TS, (M + TS - 1) / TS);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), M, N, K);

    return output;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);"

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.M = 256
        self.N = 256
        self.K = 131072 * 4

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert A.shape == (self.M, self.K), f"Expected A shape {self.M}x{self.K}, got {A.shape}"
        assert B.shape == (self.K, self.N), f"Expected B shape {self.K}x{self.N}, got {B.shape}"
        return matmul.matmul_cuda(A, B, self.M, self.N, self.K)

def get_inputs():
    A = torch.randn(256, 131072 * 4).cuda()
    B = torch.randn(131072 * 4, 256).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization needed