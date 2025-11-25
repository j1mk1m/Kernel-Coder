import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void tensor_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M_total, int K, int L) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load tiles into shared memory
        if ((m * TILE_WIDTH + tx) < K && Row < M_total) {
            sA[ty][tx] = A[Row * K + (m * TILE_WIDTH + tx)];
        } else {
            sA[ty][tx] = 0.0f;
        }
        if ((m * TILE_WIDTH + ty) < K && Col < L) {
            sB[ty][tx] = B[(m * TILE_WIDTH + ty) * L + Col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Compute tile contribution
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (Row < M_total && Col < L) {
        C[Row * L + Col] = Pvalue;
    }
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    const int M_total = N * M;
    assert(A.size(2) == B.size(0));

    auto C = torch::empty({N, M, L}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (L + TILE_WIDTH - 1) / TILE_WIDTH,
        (M_total + TILE_WIDTH - 1) / TILE_WIDTH
    );

    size_t sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    tensor_matmul_kernel<<<blocks, threads, sharedMemSize, torch::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M_total, K, L
    );

    return C;
}
"""

tensor_matmul_cpp_source = """
#include <torch/extension.h>
torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA code inline
tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_matmul = tensor_matmul  # Store the loaded CUDA module

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_cuda(A, B)