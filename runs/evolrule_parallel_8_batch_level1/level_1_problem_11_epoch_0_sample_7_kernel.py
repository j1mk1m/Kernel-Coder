import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>

__global__ void tensor_matmul_kernel(
    const float* A,
    const float* B_transposed,
    float* C,
    int B_dim,
    int I_dim,
    int J_dim,
    int L_dim,
    int K_dim) {

    int b = blockIdx.x;
    int i = blockIdx.y;
    int j = blockIdx.z;
    int k = threadIdx.x;

    if (b >= B_dim || i >= I_dim || j >= J_dim || k >= K_dim) return;

    float sum = 0.0f;

    for (int l = 0; l < L_dim; ++l) {
        int a_offset = b * I_dim * J_dim * L_dim;
        a_offset += i * J_dim * L_dim;
        a_offset += j * L_dim;
        a_offset += l;
        const float a_val = A[a_offset];

        int bk_offset = k * L_dim + l;
        const float b_val = B_transposed[bk_offset];

        sum += a_val * b_val;
    }

    int c_offset = b * I_dim * J_dim * K_dim;
    c_offset += i * J_dim * K_dim;
    c_offset += j * K_dim;
    c_offset += k;
    C[c_offset] = sum;
}

torch::Tensor tensor_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B_transposed,
    int B_dim,
    int I_dim,
    int J_dim,
    int L_dim,
    int K_dim) {

    auto C = torch::zeros({B_dim, I_dim, J_dim, K_dim}, A.options());

    const int threads = K_dim;
    const dim3 blocks(B_dim, I_dim, J_dim);
    const dim3 threads_per_block(threads, 1, 1);

    tensor_matmul_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B_transposed.data_ptr<float>(),
        C.data_ptr<float>(),
        B_dim, I_dim, J_dim, L_dim, K_dim);

    return C;
}
"""

tensor_matmul_cpp_source = """
torch::Tensor tensor_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B_transposed,
    int B_dim,
    int I_dim,
    int J_dim,
    int L_dim,
    int K_dim);
"""

# Compile the CUDA kernel
tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        B_transposed = B.t().contiguous()  # Transpose B to (K, L) for efficient access
        B_dim, I_dim, J_dim, L_dim = A.shape
        K_dim = B.size(1)
        return tensor_matmul.tensor_matmul_cuda(
            A,
            B_transposed,
            B_dim, I_dim, J_dim, L_dim, K_dim
        )