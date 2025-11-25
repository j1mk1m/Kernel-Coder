import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel implementation
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void diag_matmul_kernel(const scalar_t* __restrict__ diag, const scalar_t* __restrict__ B, scalar_t* __restrict__ out, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        out[row * M + col] = diag[row] * B[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor B) {
    const int N = diag.size(0);
    const int M = B.size(1);
    
    auto out = torch::empty({N, M}, B.options());
    
    dim3 threads(32, 8);
    dim3 blocks((M + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);
    
    AT_DISPATCH_ALL_TYPES(B.scalar_type(), "diag_matmul_cuda", ([&] {
        using scalar_t = torch::scalar_object<scalar>;
        diag_matmul_kernel<scalar_t><<<blocks, threads>>>(
            diag.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, M);
    }));
    
    return out;
}
"""

# Compile the kernel
diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources="",
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A.cuda(), B.cuda())

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def get_init_inputs():
    return []