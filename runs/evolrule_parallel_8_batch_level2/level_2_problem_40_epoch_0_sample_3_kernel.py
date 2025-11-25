import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused operations: matmul + scaling + residual addition
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* x, const float* weight, const float* bias, float scaling_factor, float* output, int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int i = idx / out_features;
    int j = idx % out_features;

    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        sum += x[i * in_features + k] * weight[j * in_features + k];
    }
    sum += bias[j];
    sum *= (scaling_factor + 1.0f);
    output[i * out_features + j] = sum;
}

torch::Tensor fused_kernel_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float scaling_factor) {
    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, device=x.device());

    const int total_elements = batch_size * out_features;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_kernel_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float scaling_factor);
"""

# Compile the fused CUDA kernel
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_kernel_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize weight and bias similar to PyTorch's Linear layer
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return fused_op.fused_kernel_cuda(x, self.weight, self.bias, self.scaling_factor)

def get_inputs():
    # Inputs should be on CUDA for the fused kernel
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]