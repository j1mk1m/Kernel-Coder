import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float A_tile[BLOCK_SIZE][K];
    __shared__ float B_tile[K][BLOCK_SIZE];

    int block_row = blockIdx.y * BLOCK_SIZE;
    int block_col = blockIdx.x * BLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float C_value = 0.0f;

    // Load A and B tiles into shared memory
    for (int k = 0; k < K; ++k) {
        // Load A_tile[ty][k]
        if (block_row + ty < M) {
            A_tile[ty][k] = A[(block_row + ty) * K + k];
        }
        // Load B_tile[k][tx]
        if (block_col + tx < N) {
            B_tile[k][tx] = B[k * N + (block_col + tx)];
        }
    }
    __syncthreads();

    // Compute the dot product
    for (int k = 0; k < K; ++k) {
        C_value += A_tile[ty][k] * B_tile[k][tx];
    }
    __syncthreads();

    // Write result
    int row = block_row + ty;
    int col = block_col + tx;
    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check dimensions
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Both A and B must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions");
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                      B.data_ptr<float>(),
                                      C.data_ptr<float>(),
                                      M, N, K);

    return C;
}
"""

matmul_cuda_header = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cuda_header,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)