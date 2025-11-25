import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int N, int M, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = blockIdx.x * BLOCK_SIZE;
    int bBegin = blockIdx.y * BLOCK_SIZE;

    float sum = 0.0f;

    for (int m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        int aEnd = min(aBegin + BLOCK_SIZE, N);
        int bEnd = min(bBegin + BLOCK_SIZE, K);

        // Load A and B into shared memory
        As[ty][tx] = (aBegin + ty < N && bBegin + tx < K) ? A[(aBegin + ty) * K + (bBegin + tx)] : 0.0f;
        Bs[ty][tx] = (bBegin + ty < K && m * BLOCK_SIZE + tx < K) ? B[(bBegin + ty) * K + (m * BLOCK_SIZE + tx)] : 0.0f;

        __syncthreads();

        // Compute C within the sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write C to device memory
    if (aBegin + ty < N && bBegin + tx < K) {
        C[(aBegin + ty) * K + (bBegin + tx)] = sum;
    }
}

torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto M = A.size(1);
    auto K = B.size(1);
    auto out = torch::zeros({N, M, K}, A.options());

    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matrix_mul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M, K);

    return out;
}
"""

matrix_mul_cpp_source = (
    "torch::Tensor matrix_mul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matrix_mul = load_inline(
    name="matrix_mul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_mul = matrix_mul

    def forward(self, A, B):
        return self.matrix_mul.matrix_mul_cuda(A, B)


if __name__ == "__main__":
    A, B = get_inputs()
    model = ModelNew().cuda()
    output = model(A.cuda(), B.cuda())
    print(output.shape)  # Should print torch.Size([16, 1024, 768])