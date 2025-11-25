import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define N 32

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int i_start = blockIdx.x * blockDim.x;
    int j_start = blockIdx.y * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load A's rows into shared memory
    if (i_start + tx < M && ty < N) {
        int i = i_start + tx;
        As[tx][ty] = A[i * N + ty];
    }

    // Load B's columns into shared memory
    if (ty < N && j_start + tx < M) {
        int j = j_start + tx;
        Bs[ty][tx] = B[ty * M + j];
    }

    __syncthreads();

    float sum = 0.0f;
    if (tx < blockDim.x && ty < blockDim.y) {
        if (i_start + tx < M && j_start + ty < M) {
            for (int k = 0; k < N; ++k) {
                sum += As[tx][k] * Bs[k][ty];
            }
            int i = i_start + tx;
            int j = j_start + ty;
            C[i * M + j] = sum;
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int N_dim = A.size(1);
    assert(N_dim == N, "This kernel requires N=32");
    auto C = torch::empty({M, M}, A.options());

    dim3 block_dim(32, 32);
    dim3 grid_dim(
        (M + block_dim.x - 1) / block_dim.x,
        (M + block_dim.y - 1) / block_dim.y
    );

    matmul_kernel<<<grid_dim, block_dim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M);

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A, B):
        return self.matmul_cuda.matmul_cuda(A, B)