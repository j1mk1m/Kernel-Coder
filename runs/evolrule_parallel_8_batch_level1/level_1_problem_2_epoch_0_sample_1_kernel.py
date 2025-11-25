import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Redefine get_inputs to return CUDA tensors
def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []

# Define the custom CUDA kernel for matrix multiplication
matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrixMultiply(
    float* C, const float* A, const float* B,
    int M, int K, int N) {

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cvalue = 0;

    // Declare shared memory outside the loop
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {

        // Load A tile into shared memory
        int rowA = by * BLOCK_SIZE + ty;
        int colA = m * BLOCK_SIZE + tx;
        As[ty][tx] = (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;

        // Load B tile into shared memory
        int rowB = m * BLOCK_SIZE + ty;
        int colB = bx * BLOCK_SIZE + tx;
        Bs[ty][tx] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;

        __syncthreads();

        // Compute the product of the tiles
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
    }

    // Write the result to global memory
    int rowC = by * BLOCK_SIZE + ty;
    int colC = bx * BLOCK_SIZE + tx;
    if (rowC < M && colC < N) {
        C[rowC * N + colC] = Cvalue;
    }
}

torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    assert(A.size(1) == B.size(0));

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    matrixMultiply<<<blocks, threads>>>(C.data_ptr<float>(),
                                       A.data_ptr<float>(),
                                       B.data_ptr<float>(),
                                       M, K, N);

    return C;
}
"""

matrix_mult_cpp_source = """
torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix multiplication
matrix_mult = load_inline(
    name="matrix_mult",
    cpp_sources=matrix_mult_cpp_source,
    cuda_sources=matrix_mult_source,
    functions=["matrix_multiply_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mult = matrix_mult

    def forward(self, A, B):
        return self.matrix_mult.matrix_multiply_cuda(A, B)