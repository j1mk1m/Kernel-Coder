import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_matmul_kernel(
    const scalar_t* A,
    const scalar_t* B_T,
    scalar_t* C,
    int M, int N
) {
    int i = blockIdx.x;
    __shared__ scalar_t A_row[32];

    int tid = threadIdx.x;
    if (tid < N) {
        A_row[tid] = A[i * N + tid];
    }
    __syncthreads();

    int num_threads = blockDim.x;
    int chunk_size = (M + num_threads - 1) / num_threads;
    int j_start = tid * chunk_size;
    int j_end = j_start + chunk_size;
    if (j_end > M) j_end = M;

    for (int j = j_start; j < j_end; ++j) {
        scalar_t result = 0.0;
        const scalar_t* B_T_row_j = B_T + j * N;

        for (int k = 0; k < N; k += 4) {
            result += A_row[k] * B_T_row_j[k];
            result += A_row[k+1] * B_T_row_j[k+1];
            result += A_row[k+2] * B_T_row_j[k+2];
            result += A_row[k+3] * B_T_row_j[k+3];
        }

        C[i * M + j] = result;
    }
}

int custom_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B_T,
    torch::Tensor C,
    int M, int N
) {
    dim3 threads(256);
    dim3 blocks(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "custom_matmul_cuda", ([&] {
        custom_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B_T.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N
        );
    }));

    return 1;
}
"""

custom_matmul_cpp_source = """
#include <torch/extension.h>
at::Tensor custom_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B_T,
    torch::Tensor C,
    int M, int N
);
"""

custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=custom_matmul_cpp_source,
    cuda_sources=custom_matmul_source,
    functions=["custom_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.M = 16384 * 2
        self.N = 16 * 2

    def forward(self, A, B):
        A = A.cuda()
        B = B.cuda()
        B_T = B.t()
        C = torch.empty(self.M, self.M, device=A.device, dtype=A.dtype)
        custom_matmul.custom_matmul_cuda(A, B_T, C, self.M, self.N)
        return C

def get_inputs():
    A = torch.rand(16384 * 2, 16 * 2)
    B = torch.rand(16 * 2, 16384 * 2)
    return [A, B]

def get_init_inputs():
    return []