import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matvecmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matvecmul_kernel(const scalar_t* __restrict__ A,
                                const scalar_t* __restrict__ B,
                                scalar_t* __restrict__ C,
                                int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ scalar_t shared_A[32][1024]; // Adjust dimensions based on block size
    __shared__ scalar_t shared_B[1024];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    scalar_t sum = 0.0;

    for (int k = 0; k < K; k += blockDim.x * gridDim.x) {
        if (row < M && k + tx < K) {
            shared_A[ty][tx] = A[row * K + k + tx];
        } else {
            shared_A[ty][tx] = 0.0;
        }
        if (k + tx < K) {
            shared_B[tx] = B[k + tx];
        } else {
            shared_B[tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_A[ty][i] * shared_B[i];
        }

        __syncthreads();
    }

    if (row < M) {
        C[row] = sum;
    }
}

torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    auto C = torch::zeros({M}, A.options());

    const int block_dim_x = 256;
    const int block_dim_y = 8;
    dim3 threads(block_dim_x, block_dim_y);
    dim3 blocks(1, (M + block_dim_y - 1) / block_dim_y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvecmul_cuda", ([&] {
        matvecmul_kernel<scalar_t><<<blocks, threads>>>(A.data_ptr<scalar_t>(),
                                                       B.data_ptr<scalar_t>(),
                                                       C.data_ptr<scalar_t>(),
                                                       M, K);
    }));

    cudaDeviceSynchronize();
    return C.view({M, 1});
}
"""

matvecmul_cpp_source = (
    "torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix-vector multiplication
matvecmul = load_inline(
    name="matvecmul",
    cpp_sources=matvecmul_cpp_source,
    cuda_sources=matvecmul_source,
    functions=["matvecmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matvecmul = matvecmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvecmul.matvecmul_cuda(A, B)

M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, 1).cuda()
    return [A, B]

def get_init_inputs():
    return []