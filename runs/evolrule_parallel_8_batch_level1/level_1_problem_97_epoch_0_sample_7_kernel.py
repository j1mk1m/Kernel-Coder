import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled dot-product attention
attention_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void scaled_dot_product_attention_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> Q,
    const torch::PackedTensorAccessor<scalar_t,4> K,
    const torch::PackedTensorAccessor<scalar_t,4> V,
    torch::PackedTensorAccessor<scalar_t,4> output,
    const int batch_size,
    const int num_heads,
    const int sequence_length,
    const int embedding_dim,
    const float scale_factor) {

    // Calculate thread and block indices
    int b = blockIdx.x / (num_heads * sequence_length);
    int h = (blockIdx.x % (num_heads * sequence_length)) / sequence_length;
    int q_idx = (blockIdx.x % (num_heads * sequence_length)) % sequence_length;
    int k_idx = threadIdx.x;

    __shared__ scalar_t shared_K[512];  // Adjust size based on sequence_length
    __shared__ scalar_t shared_V[512];  // Adjust size based on sequence_length

    scalar_t sum = 0.0;

    // Load K and V into shared memory for this block's q_idx
    if (k_idx < sequence_length) {
        shared_K[k_idx] = K[b][h][k_idx];
        shared_V[k_idx] = V[b][h][k_idx];
    }
    __syncthreads();

    // Compute Q*K^T scaled by 1/sqrt(embedding_dim)
    scalar_t dot_product = 0.0;
    for (int d = 0; d < embedding_dim; ++d) {
        dot_product += Q[b][h][q_idx][d] * shared_K[d];
    }
    dot_product *= scale_factor;

    // Softmax computation with numerical stability
    scalar_t max_val = dot_product;
    for (int k = 0; k < sequence_length; ++k) {
        if (shared_K[k] > max_val) {
            max_val = shared_K[k];
        }
    }
    scalar_t numerator = exp(dot_product - max_val);
    scalar_t denominator = 0.0;
    for (int k = 0; k < sequence_length; ++k) {
        denominator += exp(shared_K[k] - max_val);
    }
    scalar_t softmax_val = numerator / denominator;

    // Multiply by V and accumulate
    for (int d = 0; d < embedding_dim; ++d) {
        sum += softmax_val * shared_V[d];
    }

    // Write result to output
    output[b][h][q_idx] = sum;
}

torch::Tensor scaled_dot_product_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {
    
    const auto batch_size = Q.size(0);
    const auto num_heads = Q.size(1);
    const auto sequence_length = Q.size(2);
    const auto embedding_dim = Q.size(3);
    const auto scale_factor = 1.0 / sqrt(embedding_dim);

    auto output = torch::zeros({batch_size, num_heads, sequence_length, embedding_dim}, 
                              Q.options());

    dim3 blocks(batch_size * num_heads * sequence_length);
    dim3 threads(sequence_length);

    AT_DISPATCH_FLOATING_TYPES(Q.scalar_type(), "scaled_dot_product_attention_forward_cuda", ([&] {
        scaled_dot_product_attention_forward_kernel<scalar_t><<<blocks, threads>>>(
            Q.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            K.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            V.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size,
            num_heads,
            sequence_length,
            embedding_dim,
            scale_factor);
    }));

    return output;
}
"""

attention_kernel_cpp_source = """
torch::Tensor scaled_dot_product_attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V);
"""

# Compile the attention kernel
scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cpp_sources=attention_kernel_cpp_source,
    cuda_sources=attention_kernel_source,
    functions=["scaled_dot_product_attention_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.attention_kernel = scaled_dot_product_attention

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.attention_kernel.scaled_dot_product_attention_forward_cuda(Q, K, V)

def get_inputs():
    batch_size = 32
    num_heads = 32
    sequence_length = 512
    embedding_dimension = 1024
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device='cuda', dtype=torch.float16)
    return [Q, K, V]

def get_init_inputs():
    return []