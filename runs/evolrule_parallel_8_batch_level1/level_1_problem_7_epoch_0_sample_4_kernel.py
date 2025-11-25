import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 16384 * 2
N = 16384 * 2
K = 32 * 2

cuda_source = """
#include <torch/extension.h>

#define TB 16

__global__ void matrix_mult_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {

    __shared__ float shared_A[TB][TB];
    __shared__ float shared_B[TB][TB];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = block_row * TB + ty;
    int col = block_col * TB + tx;

    float sum = 0.0f;

    int num_tiles = (K + TB - 1) / TB;

    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * TB;
        int k_end = min(k_start + TB, K);

        if (ty < TB && (k_start + tx) < K) {
            shared_A[ty][tx] = A[row * K + (k_start + tx)];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        if (tx < TB && (k_start + ty) < K) {
            shared_B[ty][tx] = B[(k_start + ty) * N + col];
        } else {
            shared_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k_local = 0; k_local < TB; k_local++) {
            if (k_start + k_local < K) {
                sum += shared_A[ty][k_local] * shared_B[k_local][tx];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_mult_cuda(
    torch::Tensor A,
    torch::Tensor B) {

    int M = A.size(0);
    int K_A = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);

    if (K_A != K_B) {
        throw std::runtime_error("Incompatible matrix dimensions");
    }

    int K_val = K_A;
    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TB, TB);
    int blocks_x = (N + TB - 1) / TB;
    int blocks_y = (M + TB - 1) / TB;
    dim3 blocks(blocks_x, blocks_y);

    matrix_mult_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K_val
    );

    cudaDeviceSynchronize();

    return C;
}
"""

cpp_source = """
torch::Tensor matrix_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

matrix_mult_cuda = load_inline(
    name="matrix_mult",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matrix_mult_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_mult = matrix_mult_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_mult.matrix_mult_cuda(A, B)

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []