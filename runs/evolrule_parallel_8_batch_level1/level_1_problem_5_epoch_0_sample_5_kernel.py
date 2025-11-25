import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scalar multiplication
scalar_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void scalar_mult_kernel(const scalar_t* A, scalar_t s, scalar_t* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor scalar_mult_cuda(torch::Tensor A, float s) {
    auto size = A.numel();
    auto C = torch::empty_like(A);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "scalar_mult_cuda", ([&] {
        scalar_mult_kernel<scalar_t><<<num_blocks, block_size>>>(
            A.data_ptr<scalar_t>(), s, C.data_ptr<scalar_t>(), size);
    }));

    return C;
}
"""

scalar_mult_cpp_source = "torch::Tensor scalar_mult_cuda(torch::Tensor A, float s);"

# Compile the inline CUDA code for scalar multiplication
scalar_mult = load_inline(
    name="scalar_mult",
    cpp_sources=scalar_mult_cpp_source,
    cuda_sources=scalar_mult_source,
    functions=["scalar_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scalar_mult = scalar_mult

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_mult.scalar_mult_cuda(A, s)

def get_inputs():
    M = 16384 * 4
    N = 4096 * 4
    A = torch.rand(M, N).cuda()
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []