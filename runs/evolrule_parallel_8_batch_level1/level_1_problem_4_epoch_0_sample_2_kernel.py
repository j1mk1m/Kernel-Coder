import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matvec_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvec_kernel(const float* A, const float* B, float* out, int M, int K) {
    int i = blockIdx.x;
    if (i >= M) return;

    extern __shared__ float partial_sums[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    int chunk_size = (K + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > K) end = K;

    for (int k = start; k < end; ++k) {
        sum += A[i * K + k] * B[k];
    }

    partial_sums[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[i] = partial_sums[0];
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    assert(B.sizes() == torch::IntArrayRef({K, 1}), "B must be Kx1");

    auto out = torch::empty({M, 1}, A.options());

    const int block_size = 1024;
    int num_blocks = M;

    size_t smem_size = block_size * sizeof(float);

    matvec_kernel<<<num_blocks, block_size, smem_size, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), M, K
    );

    return out;
}
"""

matvec_cpp_source = "torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);"

matvec = load_inline(
    name="matvec",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_source,
    functions=["matvec_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matvec = matvec

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec.matvec_cuda(A, B)