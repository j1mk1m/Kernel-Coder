import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code for fused scaled dot product attention
scaled_dot_product_attention_source = """
#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUBLAS(err) do { \\
    if (err != CUBLAS_STATUS_SUCCESS) { \\
        fprintf(stderr, "CUBLAS error %d\\n", err); \\
        exit(EXIT_FAILURE); \\
    } \\
} while (0)

__global__ void row_softmax(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int cols) {
    extern __shared__ __half s_data[];
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Load the row into shared memory
    for (int j = tid; j < cols; j += blockDim.x) {
        s_data[j] = input[row_idx * cols + j];
    }
    __syncthreads();

    // Compute max
    __half max_val = __float2half(-FLT_MAX);
    for (int j = 0; j < cols; ++j) {
        if (s_data[j] > max_val) {
            max_val = s_data[j];
        }
    }
    __syncthreads();

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        float val_f = __half2float(s_data[j]);
        val_f -= __half2float(max_val);
        float exp_val = expf(val_f);
        s_data[j] = __float2half(exp_val);
        sum += exp_val;
    }
    __syncthreads();

    // Write back to output
    for (int j = tid; j < cols; j += blockDim.x) {
        float val = __half2float(s_data[j]);
        val /= sum;
        output[row_idx * cols + j] = __float2half(val);
    }
}

torch::Tensor scaled_dot_product_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {
    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int seq_q = Q.size(2);
    int d_k = Q.size(3);
    int seq_k = K.size(2);

    float scale_factor = 1.0f / sqrt(d_k);

    Q = Q.reshape({batch_size * num_heads, seq_q, d_k});
    K = K.transpose(-1, -2).reshape({batch_size * num_heads, d_k, seq_k});
    V = V.reshape({batch_size * num_heads, seq_k, d_k});

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

    // Compute QK^T
    torch::Tensor attn = torch::empty({batch_size * num_heads, seq_q, seq_k}, Q.options());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        seq_q,
        seq_k,
        d_k,
        &alpha,
        Q.data_ptr<__half>(),
        seq_q,
        K.data_ptr<__half>(),
        d_k,
        &beta,
        attn.data_ptr<__half>(),
        seq_q);

    // Apply scaling
    auto scale = torch::full({1}, scale_factor, torch::kFloat16).cuda();
    attn.mul_(scale);

    // Compute softmax
    int rows = batch_size * num_heads * seq_q;
    int cols = seq_k;
    torch::Tensor attn_softmax = torch::empty_like(attn);
    row_softmax<<<rows, 256, cols * sizeof(__half)>>>(
        attn.data_ptr<__half>(),
        attn_softmax.data_ptr<__half>(),
        cols);

    // Compute output = attn_softmax @ V
    torch::Tensor output = torch::empty(
        {batch_size * num_heads, seq_q, d_k},
        Q.options());
    cublasHgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        seq_q,
        d_k,
        seq_k,
        &alpha,
        attn_softmax.data_ptr<__half>(),
        seq_q,
        V.data_ptr<__half>(),
        seq_k,
        &beta,
        output.data_ptr<__half>(),
        seq_q);

    cublasDestroy(handle);
    output = output.view({batch_size, num_heads, seq_q, d_k});
    return output;
}
"""

scaled_dot_product_attention_cpp = (
    "torch::Tensor scaled_dot_product_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"
)

# Compile the custom CUDA code
scaled_dot_product_attention = load_inline(
    name="scaled_dot_product_attention",
    cuda_sources=scaled_dot_product_attention_source,
    cpp_sources=scaled_dot_product_attention_cpp,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention.scaled_dot_product_attention_cuda(Q, K, V)