import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused scaled dot-product attention
scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>
#include <ATen/Parallel.h>

#define BLOCK_DIM 32
#define WARP_SIZE 32
#define LOG_EPILOGUE_FRACTION 1

template <typename scalar_t>
__global__ void fused_scaled_dot_product_attention(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len,
    int emb_dim,
    float scale) {

    extern __shared__ scalar_t shared[];
    scalar_t* smem = shared;
    scalar_t* row_buf = smem;
    scalar_t* col_buf = row_buf + BLOCK_DIM * BLOCK_DIM;

    const int head_id = blockIdx.z;
    const int batch_id = blockIdx.x;
    const int head_idx = batch_id * num_heads + head_id;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Matrix multiplication Q*K^T
    scalar_t sum = 0.0;
    for (int i = 0; i < seq_len; i += BLOCK_DIM) {
        scalar_t q_val = __ldg(q + head_idx * seq_len * emb_dim + (ty + i)*emb_dim + tx);
        scalar_t k_val = __ldg(k + head_idx * seq_len * emb_dim + (tx + i)*emb_dim + ty);
        sum += q_val * k_val;
    }

    // Synchronize to ensure all threads have completed their computations
    __syncthreads();

    // Apply scaling and softmax in shared memory
    sum *= scale;
    scalar_t max_val = blockReduceMax(sum);
    __shared__ scalar_t block_max;
    if (threadIdx.x == 0) block_max = max_val;
    __syncthreads();

    sum = __expf(sum - block_max);
    scalar_t sum_exp = blockReduceSum(sum);
    __shared__ scalar_t block_sum_exp;
    if (threadIdx.x == 0) block_sum_exp = sum_exp;
    __syncthreads();

    sum = sum / block_sum_exp;

    // Matrix multiplication with V
    scalar_t output = 0.0;
    for (int i = 0; i < seq_len; i += BLOCK_DIM) {
        scalar_t v_val = __ldg(v + head_idx * seq_len * emb_dim + (tx + i)*emb_dim + ty);
        output += sum * v_val;
    }

    out[head_idx * seq_len * emb_dim + ty * seq_len + tx] = output;
}

// Helper function for block reduction
template <typename T>
__device__ T blockReduceSum(T val) {
    __shared__ volatile T shared[32];
    int lid = threadIdx.x % 32;
    int vid = threadIdx.x / 32;
    for (int delta = 16; delta > 0; delta /= 2) {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    if (lid == 0) shared[vid] = val;
    __syncthreads();
    val = (threadIdx.x < 16) ? shared[lid] : 0;
    for (int delta = 8; delta > 0; delta /= 2) {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;
}

template <typename T>
__device__ T blockReduceMax(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    T block_max = __shfl_sync(0xFFFFFFFF, val, 0);
    return block_max;
}

torch::Tensor fused_scaled_dot_product_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale) {

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto emb_dim = q.size(3);

    auto out = torch::empty({batch_size, num_heads, seq_len, emb_dim}, q.options());

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(batch_size, seq_len / BLOCK_DIM, num_heads);

    const size_t shared_mem = 2 * BLOCK_DIM * BLOCK_DIM * sizeof(float);

    fused_scaled_dot_product_attention<<<blocks, threads, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, num_heads, seq_len, emb_dim, scale
    );

    return out;
}
"""

cpp_source = "torch::Tensor fused_scaled_dot_product_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale);"

# Compile the custom CUDA kernel
fused_attention = load_inline(
    name='fused_scaled_dot_product_attention',
    cpp_sources=cpp_source,
    cuda_sources=scaled_dot_product_attention_source,
    functions=['fused_scaled_dot_product_attention_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = fused_attention

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        emb_dim = Q.size(-1)
        scale = 1.0 / (emb_dim ** 0.5)
        return self.attention.fused_scaled_dot_product_attention_cuda(Q, K, V, scale)