import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel source code for matrix multiplication with fixed dimensions
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define M 8205
#define K 2949
#define N 5921
#define TILE_DIM 16

__global__ void matmul_kernel(float* C, const float* A, const float* B) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    float Cvalue = 0.0f;

    for (int m = 0; m < (K + TILE_DIM - 1)/TILE_DIM; ++m) {
        int aRow = by * TILE_DIM + ty;
        int aCol = m * TILE_DIM + tx;
        As[ty][tx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        int bRow = m * TILE_DIM + ty;
        int bCol = bx * TILE_DIM + tx;
        Bs[ty][tx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto C = torch::empty({M, N}, A.options());

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim( (N + TILE_DIM -1)/TILE_DIM, (M + TILE_DIM -1)/TILE_DIM );

    matmul_kernel<<<gridDim, blockDim>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>());

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class CustomMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return matmul.matmul_cuda(A.cuda(), B.cuda())

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_output.cuda().matmul(B.t().cuda())
        grad_B = A.t().cuda().matmul(grad_output.cuda())
        return grad_A, grad_B

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return CustomMatmulFunction.apply(A, B)