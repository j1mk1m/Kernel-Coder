import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int TB_X, int TB_Y, int TB_K>
__global__ void tensor_matrix_mult(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N, int M, int K, int L,
    int A_stride_N, int A_stride_M, int A_stride_K,
    int B_stride_K, int B_stride_L,
    int C_stride_N, int C_stride_M, int C_stride_L
) {
    const int n = blockIdx.z;

    const int row_start = blockIdx.x * TB_X;
    const int col_start = blockIdx.y * TB_Y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float s_A[TB_X][TB_K];
    __shared__ float s_B[TB_K][TB_Y];

    float sum = 0.0f;

    for (int k_off = 0; k_off < K; k_off += TB_K) {
        if (row_start + tx < M && k_off + ty < K) {
            s_A[tx][ty] = A[ n * A_stride_N + (row_start + tx) * A_stride_M + (k_off + ty) * A_stride_K ];
        } else {
            s_A[tx][ty] = 0.0f;
        }

        if (k_off + tx < K && col_start + ty < L) {
            s_B[tx][ty] = B[ (k_off + tx) * B_stride_K + (col_start + ty) * B_stride_L ];
        } else {
            s_B[tx][ty] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TB_K; ++k) {
            sum += s_A[tx][k] * s_B[k][ty];
        }

        __syncthreads();  // Remove this line as it's unnecessary after computation
    }

    if (row_start + tx < M && col_start + ty < L) {
        const int idx = n * C_stride_N + (row_start + tx) * C_stride_M + (col_start + ty) * C_stride_L;
        C[idx] = sum;
    }
}

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    // Check dimensions and contiguity
    TORCH_CHECK(A.size(2) == B.size(0), "Incompatible dimensions");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");

    const auto C = torch::empty({N, M, L}, A.options());

    const int A_stride_N = A.stride(0);
    const int A_stride_M = A.stride(1);
    const int A_stride_K = A.stride(2);

    const int B_stride_K = B.stride(0);
    const int B_stride_L = B.stride(1);

    const int C_stride_N = C.stride(0);
    const int C_stride_M = C.stride(1);
    const int C_stride_L = C.stride(2);

    constexpr int TB_X = 16;
    constexpr int TB_Y = 16;
    constexpr int TB_K = 16;

    dim3 blocks(
        (M + TB_X - 1) / TB_X,
        (L + TB_Y - 1) / TB_Y,
        N
    );
    dim3 threads(TB_X, TB_Y);

    tensor_matrix_mult<TB_X, TB_Y, TB_K><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, K, L,
        A_stride_N, A_stride_M, A_stride_K,
        B_stride_K, B_stride_L,
        C_stride_N, C_stride_M, C_stride_L
    );

    return C;
}
"""

cpp_source = """
torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

tensor_matrix_mult = load_inline(
    name="tensor_matrix_mult",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["tensor_matrix_mult_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_matrix_mult = tensor_matrix_mult

    def forward(self, A, B):
        return self.tensor_matrix_mult.tensor_matrix_mult_cuda(A, B)