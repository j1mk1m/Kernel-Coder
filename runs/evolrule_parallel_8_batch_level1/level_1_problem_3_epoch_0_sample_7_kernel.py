import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void batched_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n
) {
    int batch = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    int row = threadIdx.y + blockIdx.z * blockDim.y;

    if (row < m && col < n) {
        scalar_t sum = 0;
        for (int e = 0; e < k; e++) {
            sum += A[batch * m * k + row * k + e] * B[batch * k * n + e * n + col];
        }
        C[batch * m * n + row * n + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    const int batch_size = A.size(0);
    const int m = A.size(1);
    const int k = A.size(2);
    const int n = B.size(2);

    auto C = torch::zeros({batch_size, m, n}, A.options());

    const int threads_x = 32;
    const int threads_y = 8;
    dim3 threads(threads_x, threads_y);

    dim3 blocks(
        batch_size,
        (n + threads_x - 1) / threads_x,
        (m + threads_y - 1) / threads_y
    );

    AT_DISPATCH_FLOATING_TYPES(A.type(), "batched_matmul_cuda", ([&] {
        batched_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            batch_size,
            m,
            k,
            n
        );
    }));

    return C;
}
"""

bmm_cpp_source = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=["-g", "-O3"],
    extra_cuda_cflags=["-g", "--use_fast_math", "-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm.batched_matmul_cuda(A, B)