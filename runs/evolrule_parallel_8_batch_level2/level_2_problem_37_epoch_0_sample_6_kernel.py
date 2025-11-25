import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish + bias addition
fused_swish_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_swish_add_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int col = idx % out_features;
    float x = input[idx];
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    float swish = x * sigmoid_x;
    float bias_val = bias[col];
    output[idx] = swish + bias_val;
}

torch::Tensor fused_swish_add_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    int total_elements = batch_size * out_features;

    auto output = torch::empty_like(input);

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_swish_add_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

fused_swish_add_cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_swish_add_bias_cuda(torch::Tensor input, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_swish_add = load_inline(
    name="fused_swish_add",
    cpp_sources=fused_swish_add_cpp_source,
    cuda_sources=fused_swish_add_source,
    functions=["fused_swish_add_bias_cuda"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.fused_swish_add = fused_swish_add  # The loaded kernel module

    def forward(self, x):
        x = self.matmul(x)
        # Apply fused Swish and bias addition
        x = self.fused_swish_add.fused_swish_add_bias_cuda(x, self.bias)
        x = self.group_norm(x)
        return x