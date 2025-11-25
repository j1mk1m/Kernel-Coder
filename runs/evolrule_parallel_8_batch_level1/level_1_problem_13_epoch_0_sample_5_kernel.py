import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
        matmul_symmetric_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrixMultiply(const float* A, const float* B, float* C, int N) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        int a_row = by * TILE_WIDTH + ty;
        int a_col = m * TILE_WIDTH + tx;
        shared_A[ty][tx] = (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.0f;

        int b_row = m * TILE_WIDTH + ty;
        int b_col = bx * TILE_WIDTH + tx;
        shared_B[ty][tx] = (b_row < N && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_symmetric_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, torch::device("cuda").dtype(torch::kFloat32));

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiply<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

        matmul_symmetric_cpp_source = (
            "torch::Tensor matmul_symmetric_cuda(torch::Tensor A, torch::Tensor B);"
        )

        self.matmul_symmetric = load_inline(
            name="matmul_symmetric",
            cpp_sources=matmul_symmetric_cpp_source,
            cuda_sources=matmul_symmetric_source,
            functions=["matmul_symmetric_cuda"],
            verbose=True,
            extra_cflags=["-std=c++14"],
            extra_ldflags=[""],
        )

    def forward(self, A, B):
        return self.matmul_symmetric.matmul_symmetric_cuda(A.cuda(), B.cuda())