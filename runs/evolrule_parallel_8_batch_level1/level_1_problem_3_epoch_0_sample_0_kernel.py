import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 32

extern "C" __global__ void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n,
    int stride_a,
    int stride_b,
    int stride_c) {

    int batch = blockIdx.z;
    int tile_row = blockIdx.x;
    int tile_col = blockIdx.y;

    if (batch >= batch_size || tile_row >= (m + TILE_WIDTH - 1)/TILE_WIDTH || tile_col >= (n + TILE_WIDTH - 1)/TILE_WIDTH) {
        return;
    }

    int row = tile_row * TILE_WIDTH + threadIdx.x;
    int col = tile_col * TILE_WIDTH + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    int output_offset = batch * stride_c + row * n + col;

    float sum = 0.0f;

    int num_chunks = (k + TILE_WIDTH -1) / TILE_WIDTH;

    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    for (int chunk = 0; chunk < num_chunks; chunk++) {

        int a_col = chunk * TILE_WIDTH + threadIdx.y;
        int a_row = row;

        if (a_col < k) {
            shared_A[threadIdx.x][threadIdx.y] = A[ batch * stride_a + a_row * k + a_col ];
        } else {
            shared_A[threadIdx.x][threadIdx.y] = 0.0f;
        }

        int b_row = chunk * TILE_WIDTH + threadIdx.x;
        int b_col = col;

        if (b_row < k) {
            shared_B[threadIdx.x][threadIdx.y] = B[ batch * stride_b + b_row * n + b_col ];
        } else {
            shared_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < TILE_WIDTH; ++kk) {
            sum += shared_A[threadIdx.x][kk] * shared_B[kk][threadIdx.y];
        }

        __syncthreads();
    }

    C[output_offset] = sum;
}

extern "C" torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check dimensions
    if (A.size(0) != B.size(0)) {
        throw std::runtime_error("Batch sizes must match");
    }
    if (A.size(2) != B.size(1)) {
        throw std::runtime_error("Matrix dimensions must agree");
    }

    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);

    auto C = torch::empty({batch_size, m, n}, A.options());

    int m_blocks = (m + TILE_WIDTH - 1) / TILE_WIDTH;
    int n_blocks = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 grid(m_blocks, n_blocks, batch_size);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    // Strides
    int stride_a = A.stride(0);  // batch stride = m*k
    int stride_b = B.stride(0);  // batch stride = k*n
    int stride_c = C.stride(0);  // batch stride = m*n

    batched_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        k,
        n,
        stride_a,
        stride_b,
        stride_c
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel failed");
    }

    return C;
}
"""

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batched_matmul_cuda = load_inline(
            name="batched_matmul",
            cuda_sources=batched_matmul_source,
            functions=["batched_matmul_cuda"],
            verbose=True,
        ).batched_matmul_cuda

    def forward(self, A, B):
        return self.batched_matmul_cuda(A, B)

def get_inputs():
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []

# Constants from the original problem
batch_size = 128
m = 128 * 4  # 512
k = 256 * 4  # 1024
n = 512 * 4  # 2048