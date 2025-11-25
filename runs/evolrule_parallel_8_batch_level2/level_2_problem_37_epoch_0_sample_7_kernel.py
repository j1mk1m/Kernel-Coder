import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

# Define the fused CUDA kernel for bias addition and Swish activation
fused_bias_add_swish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bias_add_swish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features)
        return;

    int f = idx % out_features;
    float val = input[idx] + bias[f];
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    output[idx] = val * sigmoid_val;
}

torch::Tensor fused_bias_add_swish_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto out_features = input.size(1);
    auto output = torch::empty_like(input);

    const int total_elements = batch_size * out_features;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_bias_add_swish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

fused_bias_add_swish_header = """
torch::Tensor fused_bias_add_swish_cuda(torch::Tensor input, torch::Tensor bias);
"""

# Compile the fused CUDA kernel
fused_bias_add_swish = load_inline(
    name="fused_bias_add_swish",
    cpp_sources=fused_bias_add_swish_header,
    cuda_sources=fused_bias_add_swish_source,
    functions=["fused_bias_add_swish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.fused_kernel = fused_bias_add_swish  # Loaded CUDA module

    def forward(self, x):
        x = self.matmul(x)  # Matrix multiplication with bias (from Linear layer)
        x = self.fused_kernel.fused_bias_add_swish_cuda(x, self.bias)  # Fused bias addition and Swish
        x = self.group_norm(x)  # GroupNorm (PyTorch implementation)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]