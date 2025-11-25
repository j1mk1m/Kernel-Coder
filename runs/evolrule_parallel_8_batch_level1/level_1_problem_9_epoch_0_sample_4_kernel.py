import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_kernel(const scalar_t* A, const scalar_t* B_T, scalar_t* C, int M, int N) {
    extern __shared__ scalar_t shared_A[];
    int i = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load A[i] into shared memory
    for (int k = tid; k < N; k += num_threads) {
        shared_A[k] = A[i * N + k];
    }
    __syncthreads();

    // Process j indices
    for (int j = tid; j < M; j += num_threads) {
        scalar_t sum = 0.0f;

        for (int k = 0; k < N; k += 4) {
            sum += shared_A[k] * B_T[j * N + k];
            sum += shared_A[k + 1] * B_T[j * N + k + 1];
            sum += shared_A[k + 2] * B_T[j * N + k + 2];
            sum += shared_A[k + 3] * B_T[j * N + k + 3];
        }

        C[i * M + j] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B_T, int M, int N) {
    const int threads_per_block = 256;
    const dim3 blocks(M);
    const dim3 threads(threads_per_block);

    auto C = torch::empty({M, M}, A.options());

    const size_t shared_size = N * sizeof(float);
    matmul_kernel<float><<<blocks, threads, shared_size>>>(
        A.data_ptr<float>(), B_T.data_ptr<float>(), C.data_ptr<float>(), M, N
    );

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B_T, int M, int N);
"""

matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_op = matmul_op

    def forward(self, A, B):
        # Transpose and ensure contiguous memory for B
        B_T = B.t().contiguous()
        M = A.size(0)
        N = A.size(1)
        return self.matmul_op.matmul_cuda(A, B_T, M, N)