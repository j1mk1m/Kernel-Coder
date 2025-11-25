import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom matrix multiplication kernel with shared memory and tiling
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

template <int TileDim>
__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float shared_A[TileDim][TileDim];
    __shared__ float shared_B[TileDim][TileDim];

    float acc = 0.0;

    for (int k = 0; k < (K + TileDim - 1) / TileDim; ++k) {
        // Load tiles of A and B into shared memory
        int a_row = by * TileDim + ty;
        int a_col = k * TileDim + tx;
        shared_A[ty][tx] = (a_col < K) ? A[a_row * K + a_col] : 0.0f;
        
        int b_row = k * TileDim + ty;
        int b_col = bx * TileDim + tx;
        shared_B[ty][tx] = (b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        // Compute the dot product of the current tiles
        for (int t = 0; t < TileDim; ++t) {
            if (k * TileDim + t < K) {
                acc += shared_A[ty][t] * shared_B[t][tx];
            }
        }

        __syncthreads();
    }

    // Write the computed value to C
    int c_row = by * TileDim + ty;
    int c_col = bx * TileDim + tx;
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = acc;
    }
}

torch::Tensor tiled_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M, int N, int K) {

    const int TileDim = 32; // Tuned for 1024x1024 blocks
    dim3 threads(TileDim, TileDim);
    dim3 blocks((N + TileDim - 1)/TileDim, (M + TileDim - 1)/TileDim);

    auto C = torch::empty({M, N}, A.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "tiled_matmul_cuda", ([&] {
        tiled_matmul_kernel<TileDim><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}
"""

matmul_kernel_cpp = """
torch::Tensor tiled_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M, int N, int K);
"""

# Compile the custom CUDA kernel
tiled_matmul = load_inline(
    name="tiled_matmul",
    cpp_sources=matmul_kernel_cpp,
    cuda_sources=matmul_kernel_source,
    functions=["tiled_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tiled_matmul = tiled_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Transpose inputs as per original model's requirement
        A = A.t().contiguous()
        B = B.t().contiguous()
        # Move tensors to CUDA
        A = A.cuda()
        B = B.cuda()

        # Get dimensions
        M = A.size(0)
        K = A.size(1)
        N = B.size(1)
        
        # Call the custom CUDA kernel
        return self.tiled_matmul.tiled_matmul_cuda(A, B, M, N, K).cpu()

# Preserve original get_inputs and get_init_inputs
def get_inputs():
    A = torch.rand(K, M)  # Original dimensions: M=2048, K=8192, N=4096
    B = torch.rand(N, K)
    return [A, B]

def get_init_inputs():
    return []