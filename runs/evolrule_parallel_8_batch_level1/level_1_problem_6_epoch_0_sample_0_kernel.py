import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

M = 256
N = 256
K = 131072 * 4

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TK 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TK];
    __shared__ float Bs[TK][TILE_WIDTH];

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    // Compute the global row and column indices in C
    unsigned int Row = by * TILE_WIDTH + ty;
    unsigned int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int k = 0; k < K; k += TK) {
        // Load A tile into shared memory
        if ((by * TILE_WIDTH + ty) < M && (k + tx) < K) {
            As[ty][tx] = A[(by * TILE_WIDTH + ty) * K + (k + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        if ((k + tx) < K && (bx * TILE_WIDTH + ty) < N) {
            Bs[tx][ty] = B[(k + tx) * N + (bx * TILE_WIDTH + ty)];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product of the current tiles
        for (int i = 0; i < TK; ++i) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write the result to C
    if (Row < M && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check dimensions
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed