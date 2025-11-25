import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
custom_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    extern __shared__ float shared_sum[];
    int tid = threadIdx.x;

    int chunk_size = (K + blockDim.x - 1) / blockDim.x;

    float local_sum = 0.0f;

    for (int k = tid * chunk_size; k < (tid+1)*chunk_size; ++k) {
        if (k < K) {
            float a_val = A[k * M + blockIdx.x]; // A is K rows × M columns → A[k][i] is A[k*M +i]
            float b_val = B[blockIdx.y * K + k]; // B is N rows × K columns → B[j][k] is B[j*K +k]
            local_sum += a_val * b_val;
        }
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x/2; s > 0; s >>=1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[blockIdx.x * N + blockIdx.y] = shared_sum[0];
    }
}

torch::Tensor custom_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M, int K, int N
) {
    const int threadsPerBlock = 256;
    const dim3 blocksPerGrid(M, N); // Each block handles (i,j)
    const size_t sharedSize = threadsPerBlock * sizeof(float);

    auto C = torch::empty({M, N}, A.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "custom_matmul_cuda", ([&] {
        custom_matmul<<<blocksPerGrid, threadsPerBlock, sharedSize, at::cuda::getCurrentCUDAStream()>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N
        );
    }));

    return C;
}
"""

custom_matmul_cpp_source = (
    "torch::Tensor custom_matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N);"
)

# Compile the inline CUDA code for the custom matrix multiplication
custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=custom_matmul_cpp_source,
    cuda_sources=custom_matmul_source,
    functions=["custom_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_matmul = custom_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M = A.size(1)
        K = A.size(0)
        N = B.size(0)
        return self.custom_matmul.custom_matmul_cuda(A, B, M, K, N)

def get_inputs():
    K = 4096 * 2
    M = 1024 * 2
    N = 2048 * 2
    A = torch.rand(K, M).cuda()
    B = torch.rand(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []