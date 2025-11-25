import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 4D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tensor_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int B_dim,
    const int I_dim,
    const int J_dim,
    const int L_dim,
    const int K_dim
) {
    int batch = blockIdx.z;
    int i = blockIdx.x * blockDim.y + threadIdx.y;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I_dim || j >= J_dim) return;

    scalar_t sum = 0;
    for (int l = 0; l < L_dim; ++l) {
        sum += A[batch * I_dim * J_dim * L_dim + i * J_dim * L_dim + j * L_dim + l] *
               B[l * K_dim + threadIdx.z];
    }
    C[batch * I_dim * J_dim * K_dim + i * J_dim * K_dim + j * K_dim + threadIdx.z] = sum;
}

torch::Tensor tensor_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    const int B_dim = A.size(0);
    const int I_dim = A.size(1);
    const int J_dim = A.size(2);
    const int L_dim = A.size(3);
    const int K_dim = B.size(1);

    auto C = torch::zeros({B_dim, I_dim, J_dim, K_dim}, A.options());

    const dim3 threads(16, 16, 1); // Threads per block (x,y,z)
    const dim3 blocks(
        (J_dim + threads.x - 1) / threads.x,
        (I_dim + threads.y - 1) / threads.y,
        B_dim * K_dim // z-dimension handles batch and K dimensions
    );

    // Launch kernel for all K dimensions in parallel using z-threads
    AT_DISPATCH_FLOATING_TYPES(A.type(), "tensor_matmul_cuda", ([&] {
        tensor_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            B_dim, I_dim, J_dim, L_dim, K_dim
        );
    }));

    cudaDeviceSynchronize();
    return C;
}
"""

tensor_matmul_cpp_source = (
    "torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
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
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_cuda(A, B)