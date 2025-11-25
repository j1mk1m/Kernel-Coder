import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH BLOCK_SIZE

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float shared_A[BLOCK_SIZE][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][BLOCK_SIZE];

    int block_row = blockIdx.y * BLOCK_SIZE;
    int block_col = blockIdx.x * BLOCK_SIZE;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    float sum = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE_WIDTH) {
        int a_row = block_row + ty;
        int a_col = tile_k + tx;
        float a_val = (a_col < K) ? A[a_row * K + a_col] : 0.0f;

        int b_row = tile_k + ty;
        int b_col = block_col + tx;
        float b_val = (b_row < K) ? B[b_row * N + b_col] : 0.0f;

        shared_A[ty][tx] = a_val;
        shared_B[ty][tx] = b_val;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    int c_row = block_row + ty;
    int c_col = block_col + tx;
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K_A = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);
    assert(K_A == K_B && "Matrix dimensions must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K_A, N
    );

    return C;
}
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-arch=sm_70"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A, B)