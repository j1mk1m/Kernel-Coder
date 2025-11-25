import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

lower_tri_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void lower_tri_mult_kernel(
    const scalar_t *A,
    const scalar_t *B,
    scalar_t *C,
    int N) {

    extern __shared__ scalar_t shared_row[];

    int i = blockIdx.x;
    if (i >= N) return;

    int tid = threadIdx.x;

    // Load row i into shared memory
    for (int k = tid; k < N; k += blockDim.x) {
        shared_row[k] = A[i * N + k];
    }
    __syncthreads();

    // Compute for j
    int j_start = tid;
    int stride = blockDim.x;
    for (int j = j_start; j <= i; j += stride) {
        scalar_t sum = 0;
        for (int k = j; k <= i; ++k) {
            sum += shared_row[k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

extern "C" {

    torch::Tensor lower_tri_mult_cuda(torch::Tensor A, torch::Tensor B) {
        const int N = A.size(0);
        auto C = torch::zeros_like(A);
        
        int threads_per_block = 256;
        int blocks = N;

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "lower_tri_mult_cuda", ([&] {
            const int shared_size = N * sizeof(scalar_t);
            lower_tri_mult_kernel<scalar_t><<<blocks, threads_per_block, shared_size>>>(
                A.data<scalar_t>(),
                B.data<scalar_t>(),
                C.data_ptr<scalar_t>(),
                N
            );
        }));

        cudaDeviceSynchronize();
        return C;
    }
}
"""

lower_tri_mult_cpp_source = """
torch::Tensor lower_tri_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the CUDA code
lower_tri_mult = load_inline(
    name="lower_tri_mult",
    cpp_sources=lower_tri_mult_cpp_source,
    cuda_sources=lower_tri_mult_source,
    functions=["lower_tri_mult_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.lower_tri_mult = lower_tri_mult

    def forward(self, A, B):
        return self.lower_tri_mult.lower_tri_mult_cuda(A, B)