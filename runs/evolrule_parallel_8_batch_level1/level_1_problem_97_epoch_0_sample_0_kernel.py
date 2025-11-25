import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scaled dot-product attention
scaled_dot_attn_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void fused_scaled_dot_attn_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int emb_dim,
    const float scaling_factor) {

    extern __shared__ char shared_mem[];
    scalar_t* shared_k = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* shared_v = shared_k + seq_len * emb_dim;

    int batch = blockIdx.x;
    int head = blockIdx.y;
    int q_pos = threadIdx.x;

    // Load K and V into shared memory
    for (int idx = threadIdx.x; idx < seq_len * emb_dim; idx += blockDim.x) {
        shared_k[idx] = k[batch * num_heads * seq_len * emb_dim +
                         head * seq_len * emb_dim +
                         idx];
        shared_v[idx] = v[batch * num_heads * seq_len * emb_dim +
                         head * seq_len * emb_dim +
                         idx];
    }
    __syncthreads();

    // Compute attention scores for this query position
    scalar_t scores[seq_len];
    for (int k = 0; k < seq_len; ++k) {
        scalar_t score = 0.0;
        for (int d = 0; d < emb_dim; ++d) {
            score += q[batch * num_heads * seq_len * emb_dim +
                      head * seq_len * emb_dim +
                      q_pos * emb_dim + d] *
                    shared_k[k * emb_dim + d];
        }
        score *= scaling_factor;
        scores[k] = score;
    }
    __syncthreads();

    // Compute maximum score for this query
    scalar_t max_score = -INFINITY;
    for (int k = 0; k < seq_len; ++k) {
        if (scores[k] > max_score) {
            max_score = scores[k];
        }
    }
    __syncthreads();

    // Compute numerator and denominator
    scalar_t numerator = expf(float(scores[q_pos] - max_score));
    scalar_t denominator = 0.0f;
    for (int k = 0; k < seq_len; ++k) {
        denominator += expf(float(scores[k] - max_score));
    }
    __syncthreads();

    // Compute the weighted sum with V
    scalar_t output_val[emb_dim];
    for (int d = 0; d < emb_dim; ++d) {
        output_val[d] = 0.0;
        for (int k = 0; k < seq_len; ++k) {
            output_val[d] += (expf(float(scores[k] - max_score)) / denominator) *
                            shared_v[k * emb_dim + d];
        }
    }

    // Write the result to the output tensor
    int out_offset = batch * num_heads * seq_len * emb_dim +
                     head * seq_len * emb_dim +
                     q_pos * emb_dim;
    for (int d = 0; d < emb_dim; ++d) {
        out[out_offset + d] = output_val[d];
    }
}

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V) {

    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int emb_dim = Q.size(3);
    const float scaling_factor = 1.0f / sqrt(emb_dim);

    auto output = torch::empty_like(Q);

    dim3 blocks(batch_size, num_heads);
    dim3 threads(seq_len);

    // Calculate shared memory size for K and V (in bytes)
    size_t smem_size = 2 * seq_len * emb_dim * sizeof(float); // Use __half for FP16

    AT_DISPATCH_FLOATING_TYPES(Q.scalar_type(), "scaled_dot_product_attention_cuda", ([&] {
        fused_scaled_dot_attn_kernel<scalar_t><<<blocks, threads, smem_size, at::cuda::getCurrentCUDAStream()>>>(
            Q.data<scalar_t>(),
            K.data<scalar_t>(),
            V.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            emb_dim,
            scaling_factor
        );
    }));

    return output;
}
"""

# Compile the CUDA kernel
scaled_dot_attn = load_inline(
    name="scaled_dot_attn",
    cpp_sources="",
    cuda_sources=scaled_dot_attn_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_80']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.attention = scaled_dot_attn

    def forward(self, Q, K, V):
        return self.attention.scaled_dot_product_attention_cuda(Q, K, V)