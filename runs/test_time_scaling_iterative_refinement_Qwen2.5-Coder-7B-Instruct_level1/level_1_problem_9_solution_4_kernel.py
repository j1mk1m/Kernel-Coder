import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int aBegin = M * K * by;
    int aEnd = aBegin + M * K;
    int aStep = K;
    int bBegin = K * N * bx;
    int bEnd = bBegin + K * N;
    int bStep = N;

    float Cvalue = 0;

    for (int m = 0; m < M; m += TILE_WIDTH) {
        for (int k = 0; k < K; k += TILE_WIDTH) {
            As[ty][tx] = 0;
            Bs[ty][tx] = 0;

            // Preload data from global memory to shared memory
            if ((m + ty) < M && (k + tx) < K) {
                As[ty][tx] = A[aBegin + m * K + k + tx];
            }
            if ((k + ty) < K && (b + k * N + tx) < N) {
                Bs[ty][tx] = B[bBegin + k * N + tx];
            }

            __syncthreads();

            // Perform computation
            for (int kk = 0; kk < TILE_WIDTH; ++kk) {
                Cvalue += As[ty][kk] * Bs[kk][tx];
            }

            __syncthreads();
        }
    }

    if ((blockIdx.y * blockDim.y + threadIdx.y) < M && (blockIdx.x * blockDim.x + threadIdx.x) < N) {
        C[blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x + threadIdx.x] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto out = torch::zeros({M, N}, A.options());

    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), M, N, K);

    return out;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A, B)


# Example usage
if __name__ == "__main__":
    A = torch.rand(16384 * 2, 16 * 2).cuda()
    B = torch.rand(16 * 2, 16384 * 2).cuda()

    model_new = ModelNew().cuda()
    result = model_new(A, B)
    print(result.shape)  # Should be (16384 * 2, 16384 * 2)