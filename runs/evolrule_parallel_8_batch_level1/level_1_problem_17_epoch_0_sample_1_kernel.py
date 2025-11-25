import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C,
                             int M, int N, int K) {
    __shared__ float shared_A[BLOCK_SIZE][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * BLOCK_SIZE;
    int block_col = blockIdx.x * BLOCK_SIZE;

    int row = block_row + ty;
    int col = block_col + tx;

    float Cvalue = 0.0f;

    for (int k_tile = 0; k_tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; k_tile++) {
        int k_start = k_tile * TILE_WIDTH;

        // Load A tile into shared memory
        if (k_start + tx < K && (block_row + ty) < M) {
            shared_A[ty][tx] = A[(block_row + ty) * K + (k_start + tx)];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        // Load B tile into shared memory
        if (k_start + ty < K && (block_col + tx) < N) {
            shared_B[ty][tx] = B[(block_col + tx) * K + (k_start + ty)];
        } else {
            shared_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product for this tile
        for (int i = 0; i < TILE_WIDTH; i++) {
            Cvalue += shared_A[ty][i] * shared_B[i][tx];
        }

        // Removed unnecessary __syncthreads() here
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K_A = A.size(1);
    int N = B.size(0);
    int K_B = B.size(1);
    if (K_A != K_B) {
        throw std::runtime_error("Incompatible matrix dimensions for multiplication");
    }

    int K = K_A;
    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
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
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda()
        B = B.cuda()
        return matmul_cuda(A, B)

# The get_inputs and get_init_inputs functions remain unchanged as per the original code
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []