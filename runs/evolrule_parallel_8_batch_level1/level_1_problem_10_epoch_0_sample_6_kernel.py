import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tensor_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define N 16
#define M 1024
#define K 2048
#define L 768

__global__ void tensor_matmul_kernel(
    const float* A,
    const float* B,
    float* out
) {
    int block_n = blockIdx.x;
    int block_m = blockIdx.y;
    if (block_n >= N || block_m >= M) return;

    int l = threadIdx.x;
    if (l >= L) return;

    extern __shared__ float shared_A[];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;  // L threads per block.
    int num_elements = K;
    for (int k = tid; k < num_elements; k += num_threads) {
        int a_offset = block_n * M * K + block_m * K + k;
        shared_A[k] = A[a_offset];
    }

    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = shared_A[k];
        int b_offset = k * L + l;
        float b = B[b_offset];
        sum += a * b;
    }

    int out_offset = block_n * M * L + block_m * L + l;
    out[out_offset] = sum;
}

torch::Tensor tensor_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
) {
    auto out = torch::empty({N, M, L}, A.options());

    A = A.contiguous();
    B = B.contiguous();

    dim3 threads(L);
    dim3 blocks(N, M);

    size_t sharedMemSize = K * sizeof(float);

    tensor_matmul_kernel<<<blocks, threads, sharedMemSize, torch::cuda::current_stream()>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>()
    );

    return out;
}
"""

tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources="",
    cuda_sources=tensor_matmul_source,
    functions=["tensor_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return tensor_matmul.tensor_matmul_cuda(A, B)