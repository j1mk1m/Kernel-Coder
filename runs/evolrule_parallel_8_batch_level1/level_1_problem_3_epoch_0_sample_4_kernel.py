import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TS 16
#define BK 16

__global__ void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n) {

    int batch = blockIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.z;

    int cRow = blockRow * TS;
    int cCol = blockCol * TS;

    if (cRow >= m || cCol >= n) {
        return;
    }

    int row_in_block = threadIdx.y;
    int col_in_block = threadIdx.x;

    __shared__ float shared_A[TS][BK];
    __shared__ float shared_B[BK][TS];

    float acc = 0.0f;

    for (int chunk = 0; chunk < (k + BK - 1) / BK; chunk++) {
        int a_col_start = chunk * BK;
        int b_row_start = chunk * BK;

        // Load A tile into shared memory
        if (a_col_start + col_in_block < k) {
            int a_row = cRow + row_in_block;
            int a_col = a_col_start + col_in_block;
            shared_A[row_in_block][col_in_block] = A[batch * m * k + a_row * k + a_col];
        } else {
            shared_A[row_in_block][col_in_block] = 0.0f;
        }

        // Load B tile into shared memory
        if (b_row_start + row_in_block < k) {
            int b_row = b_row_start + row_in_block;
            int b_col = cCol + col_in_block;
            shared_B[row_in_block][col_in_block] = B[batch * k * n + b_row * n + b_col];
        } else {
            shared_B[row_in_block][col_in_block] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product for this chunk
        for (int i = 0; i < BK; i++) {
            acc += shared_A[row_in_block][i] * shared_B[i][col_in_block];
        }

        __syncthreads();
    }

    // Write the result
    int c_row = cRow + row_in_block;
    int c_col = cCol + col_in_block;
    if (c_row < m && c_col < n) {
        int idx = batch * m * n + c_row * n + c_col;
        C[idx] = acc;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k_A = A.size(2);
    int k_B = B.size(1);
    int n_out = B.size(2);

    TORCH_CHECK(B.size(0) == batch_size, "Batch sizes must match");
    TORCH_CHECK(k_A == k_B, "Input dimensions must match");

    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device());
    auto C = torch::empty({batch_size, m, n_out}, options);

    int tiles_m = (m + TS - 1) / TS;
    int tiles_n = (n_out + TS - 1) / TS;
    dim3 grid(batch_size, tiles_m, tiles_n);
    dim3 block(TS, TS);

    batched_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        k_A,
        n_out
    );

    return C;
}
"""

batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return batched_matmul.batched_matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []