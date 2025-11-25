import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tensor_matrix_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tensor_matrix_mult_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ out,
    int B_dim,
    int I_dim,
    int J_dim,
    int L_dim,
    int K_dim
) {
    int block_idx = blockIdx.x;
    int b = block_idx / (I_dim * J_dim);
    int rem = block_idx % (I_dim * J_dim);
    int i = rem / J_dim;
    int j = rem % J_dim;

    int k = threadIdx.x;
    if (k >= K_dim) return;

    scalar_t sum = 0.0;
    for (int l = 0; l < L_dim; ++l) {
        int A_offset = b * I_dim * J_dim * L_dim + i * J_dim * L_dim + j * L_dim + l;
        int B_offset = l * K_dim + k;
        sum += A[A_offset] * B[B_offset];
    }

    int out_offset = b * I_dim * J_dim * K_dim + i * J_dim * K_dim + j * K_dim + k;
    out[out_offset] = sum;
}

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int B_dim = A.size(0);
    const int I_dim = A.size(1);
    const int J_dim = A.size(2);
    const int L_dim = A.size(3);
    const int K_dim = B.size(1);

    auto out = torch::zeros({B_dim, I_dim, J_dim, K_dim}, A.options());

    dim3 blocks(B_dim * I_dim * J_dim);
    dim3 threads(K_dim);

    AT_ASSERT(A.device().is_cuda() && B.device().is_cuda());

    tensor_matrix_mult_kernel<float><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        B_dim, I_dim, J_dim, L_dim, K_dim
    );

    cudaDeviceSynchronize();
    return out;
}
"""

tensor_matrix_mult_cpp_source = """
#include <torch/extension.h>

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
tensor_matrix_mult = load_inline(
    name="tensor_matrix_mult",
    cpp_sources=tensor_matrix_mult_cpp_source,
    cuda_sources=tensor_matrix_mult_source,
    functions=["tensor_matrix_mult_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matrix_mult = tensor_matrix_mult

    def forward(self, A, B):
        return self.tensor_matrix_mult.tensor_matrix_mult_cuda(A, B)