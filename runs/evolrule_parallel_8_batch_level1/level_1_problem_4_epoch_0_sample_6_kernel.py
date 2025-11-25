import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matvec_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvec_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K
) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float sum = 0.0f;

    for (int k = tid; k < K; k += num_threads) {
        sum += A[row * K + k] * B[k];
    }

    __shared__ float shared_sums[256]; // blockDim.x is 256

    shared_sums[tid] = sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
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
    const int M = A.size(0);
    const int K = A.size(1);
    const int block_size = 256;
    const dim3 blocks(M);
    const dim3 threads(block_size);

    auto C = torch::empty({M}, A.options());

    matvec_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C.view({M, 1});
}
"""

matvec_header = "torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);"

matvec = load_inline(
    name="matvec",
    cpp_sources=matvec_header,
    cuda_sources=matvec_source,
    functions=["matvec_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matvec = matvec

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec.matvec_cuda(A, B)

M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, 1).cuda()
    return [A, B]

def get_init_inputs():
    return []