import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void diag_matmul_kernel(const scalar_t* __restrict__ A,
                                  const scalar_t* __restrict__ B,
                                  scalar_t* __restrict__ C,
                                  const int N,
                                  const int M) {
    // Each thread is responsible for one element of C (each element in B's row)
    // Index calculations:
    // Thread ID: tx (x-dimension)
    // Block ID: bx (y-dimension)
    // Element (bx, tx) in C

    const int tx = threadIdx.x + blockDim.x * blockIdx.x;
    const int bx = blockIdx.y;

    if (bx < N && tx < M) {
        const int index = bx * M + tx;
        C[index] = A[bx] * B[index];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check dimensions
    const int N = A.size(0);
    const int M = B.size(1);
    assert(A.dim() == 1 && B.dim() == 2);
    assert(A.size(0) == B.size(0));

    // Output tensor
    auto C = torch::empty({N, M}, B.options());

    // Define grid and block dimensions
    const int threads_per_block = 256;
    dim3 blocks((M + threads_per_block - 1) / threads_per_block, N);
    dim3 threads(threads_per_block, 1);

    // Launch kernel
    AT_DISPATCH_ALL_TYPES(A.scalar_type(), "diag_matmul_cuda", ([&] {
        diag_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N,
            M);
    }));

    return C;
}
"""

diag_matmul_cpp_source = """
torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA kernel
diag_matmul = load_inline(
    name="diagmatmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True,
    extra_cflags=["-D_USE_CUDNN", "-D_USE_FP16"],
    extra_cuda_cflags=["-lineinfo", "--expt-extended-lambda"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)