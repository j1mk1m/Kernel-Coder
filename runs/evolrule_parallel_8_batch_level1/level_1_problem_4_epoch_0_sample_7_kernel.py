import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matvec_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm> // for std::min

template <typename T>
__global__ void matvec_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    T sum = 0.0;

    int chunk_size = (K + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = (start + chunk_size > K) ? K : start + chunk_size;

    for (int k = start; k < end; ++k) {
        sum += A[row * K + k] * B[k];
    }

    __shared__ T shared_sums[256];
    shared_sums[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[row] = shared_sums[0];
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    // Check dimensions
    int M = A.size(0);
    int K_A = A.size(1);
    int K_B = B.size(0);
    int N_B = B.size(1);
    if (K_A != K_B || N_B != 1) {
        TORCH_CHECK(false, "Invalid dimensions");
    }

    auto C = torch::empty({M, 1}, A.options());
    const int block_size = 256;
    dim3 blocks(M);
    dim3 threads(block_size);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_cuda", ([&] {
        matvec_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K_A
        );
    }));

    return C;
}
"""

matvec_cpp_source = """
torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);
"""

matvec_ops = load_inline(
    name="matvec_cuda",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_source,
    functions=["matvec_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matvec_cuda = matvec_ops

    def forward(self, A, B):
        return self.matvec_cuda.matvec_cuda(A, B)

def get_inputs():
    # Move inputs to CUDA
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, 1).cuda()
    return [A, B]

def get_init_inputs():
    return []