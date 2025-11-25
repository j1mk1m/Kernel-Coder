import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom fused CUDA kernel for Gemm, Divide, Sum, and Scaling
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_div_sum_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float scaling_factor,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size
) {
    int batch = blockIdx.x;
    float sum = 0.0;
    for (int i = 0; i < input_size; ++i) {
        float val = input[batch * input_size + i] * weight[i];  // Gemm
        sum += val / 2.0f;  // Divide and accumulate sum
    }
    output[batch] = sum * scaling_factor;  // Scaling and store
}

torch::Tensor fused_gemm_div_sum_scale(
    torch::Tensor input,
    torch::Tensor weight,
    float scaling_factor
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);  // weight is (hidden_size, input_size)

    auto output = torch::empty({batch_size, 1}, device input.device(), dtype input.dtype());

    const int threads = 1;  // Each thread handles a batch element
    const int blocks = batch_size;

    fused_gemm_div_sum_scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_gemm_div_sum_scale(torch::Tensor input, torch::Tensor weight, float scaling_factor);"
)

# Compile the fused kernel
fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_gemm_div_sum_scale"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_kernel = fused_kernel

    def forward(self, x):
        return self.fused_kernel.fused_gemm_div_sum_scale(x, self.weight, self.scaling_factor)

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]

batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5