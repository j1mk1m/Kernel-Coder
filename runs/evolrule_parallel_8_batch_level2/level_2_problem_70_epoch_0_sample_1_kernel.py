import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel
fused_gemm_sigmoid_residual_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_gemm_sigmoid_residual_add_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float scaling_factor,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= batch_size * hidden_size) return;

    int batch = thread_idx / hidden_size;
    int hidden = thread_idx % hidden_size;

    float gemm_out = 0.0f;
    for (int i = 0; i < input_size; ++i) {
        gemm_out += input[batch * input_size + i] * weight[hidden * input_size + i];
    }
    gemm_out += bias[hidden];

    float sigmoid_val = 1.0f / (1.0f + expf(-gemm_out));
    float scaled_sigmoid = scaling_factor * sigmoid_val;
    output[thread_idx] = scaled_sigmoid + gemm_out;
}

torch::Tensor fused_gemm_sigmoid_residual_add(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    int batch_size,
    int input_size,
    int hidden_size
) {
    auto output = torch::empty({batch_size, hidden_size}, torch::dtype(input.dtype()).device(input.device()));
    
    const int threads_per_block = 256;
    const int blocks = (batch_size * hidden_size + threads_per_block - 1) / threads_per_block;

    fused_gemm_sigmoid_residual_add_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

fused_gemm_sigmoid_residual_add_header = """
torch::Tensor fused_gemm_sigmoid_residual_add(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor,
    int batch_size,
    int input_size,
    int hidden_size
);
"""

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        
        # Initialize weights and bias similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(input_size)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Load the fused CUDA kernel
        self.fused_op = load_inline(
            name="fused_gemm_sigmoid_residual_add",
            cpp_sources=fused_gemm_sigmoid_residual_add_header,
            cuda_sources=fused_gemm_sigmoid_residual_add_source,
            functions=["fused_gemm_sigmoid_residual_add"],
            verbose=True
        )

    def forward(self, x):
        batch_size = x.size(0)
        input_size = x.size(1)
        hidden_size = self.weight.size(0)
        return self.fused_op(
            x,
            self.weight,
            self.bias,
            self.scaling_factor,
            batch_size,
            input_size,
            hidden_size
        )