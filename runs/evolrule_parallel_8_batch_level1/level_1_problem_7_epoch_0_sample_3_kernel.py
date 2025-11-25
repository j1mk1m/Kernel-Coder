import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for matrix multiplication optimized for small K
matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void tiled_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                            int M, int N, int K) {
    __shared__ float shared_A[TILE_M][TILE_K];
    __shared__ float shared_B[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc[TILE_M][TILE_N];
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k_block = 0; k_block < (K + TILE_K - 1) / TILE_K; ++k_block) {
        // Load tiles into shared memory
        if (ty < TILE_M && tx < K) {
            int a_row = blockIdx.y * TILE_M + ty;
            int a_col = k_block * TILE_K + tx;
            shared_A[ty][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        if (tx < TILE_N && ty < K) {
            int b_row = k_block * TILE_K + ty;
            int b_col = blockIdx.x * TILE_N + tx;
            shared_B[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        __syncthreads();

        // Compute tile contributions
        for (int k = 0; k < TILE_K; ++k) {
            float a_val = shared_A[ty][k];
            float b_val = shared_B[k][tx];
            for (int i = 0; i < TILE_M; ++i) {
                for (int j = 0; j < TILE_N; ++j) {
                    acc[i][j] += a_val * b_val;
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    for (int i = 0; i < TILE_M; ++i) {
        for (int j = 0; j < TILE_N; ++j) {
            int row = blockIdx.y * TILE_M + i;
            int col = blockIdx.x * TILE_N + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// Launch configuration and kernel invocation
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::empty({M, N}, A.options());

    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 32; // Align with K=64 (since K=32*2)

    dim3 threads(TILE_N, TILE_M);
    dim3 blocks((N + TILE_N - 1)/TILE_N, (M + TILE_M - 1)/TILE_M);

    tiled_matmul<TILE_M, TILE_N, TILE_K><<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), 
                                                             C.data_ptr<float>(), M, N, K);

    cudaDeviceSynchronize();
    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-DUSE_CUDA"],
    extra_ldflags=["-lcudart"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_cuda  # Load the compiled kernel

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device
        A = A.cuda()
        B = B.cuda()
        return self.matmul.matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []