import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tensor_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define L 256
#define K 768
#define B_DIM 8
#define I_DIM 256
#define J_DIM 512

__global__ void tensor_matrix_mult_kernel(
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> B_t,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output
) {
    int block_idx = blockIdx.x;
    int b = block_idx / (I_DIM * J_DIM);
    int remainder = block_idx % (I_DIM * J_DIM);
    int i = remainder / J_DIM;
    int j = remainder % J_DIM;

    int k = threadIdx.x;

    __shared__ float shared_A[L];

    if (threadIdx.x < L) {
        shared_A[threadIdx.x] = A[b][i][j][threadIdx.x];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        sum += shared_A[l] * B_t[k][l];
    }

    output[b][i][j][k] = sum;
}

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B_t) {
    const int B = B_DIM;
    const int I = I_DIM;
    const int J = J_DIM;
    const int K = K;

    auto output = torch::empty({B, I, J, K}, A.options());

    const int num_blocks = B * I * J;
    const int threads_per_block = K;

    tensor_matrix_mult_kernel<<<num_blocks, threads_per_block>>>(
        A.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        B_t.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        output.packed_accessor32<float,4,torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the CUDA code
tensor_mult_cuda = load_inline(
    name="tensor_mult_cuda",
    cuda_sources=tensor_mult_source,
    functions=["tensor_matrix_mult_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_mult_cuda = tensor_mult_cuda

    def forward(self, A, B):
        B_t = B.t()  # Transpose B to (K, L) for optimized memory access
        return self.tensor_mult_cuda.tensor_matrix_mult_cuda(A, B_t)