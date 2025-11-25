import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matvec_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvec_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ result,
    int K) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float partial_sums[256];

    float sum = 0.0f;
    const int chunk_size = K / 256;

    int start = tid * chunk_size;
    int end = (tid + 1) * chunk_size;

    for (int k = start; k < end; ++k) {
        sum += A[row * K + k] * B[k];
    }

    partial_sums[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[row] = partial_sums[0];
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    auto result = torch::empty({M}, A.options());

    const int threads_per_block = 256;
    const dim3 blocks(M);
    const dim3 threads(threads_per_block);

    matvec_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), result.data_ptr<float>(), K);

    return result.view({M, 1});
}
"""

matvec_h_source = """
torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);
"""

matvec_cuda_mod = load_inline(
    name="matvec_cuda",
    cpp_sources=matvec_h_source,
    cuda_sources=matvec_source,
    functions=["matvec_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matvec_cuda_mod.matvec_cuda(A, B)