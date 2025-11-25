import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for GEMM + scale + LeakyReLU
fused_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float multiplier,
    float negative_slope,
    float* out,
    int batch_size,
    int in_features,
    int out_features
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_features) return;

    int batch_idx = tid / out_features;
    int j = tid % out_features;

    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += x[batch_idx * in_features + i] * weight[j * in_features + i];
    }
    sum += bias[j];
    sum *= multiplier;
    out[tid] = (sum > 0) ? sum : sum * negative_slope;
}

torch::Tensor fused_gemm_scale_leaky_cuda(torch::Tensor x,
                                          torch::Tensor weight,
                                          torch::Tensor bias,
                                          float multiplier,
                                          float negative_slope) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, x.options());

    int num_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_gemm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        multiplier,
        negative_slope,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_gemm_cpp_source = (
    "torch::Tensor fused_gemm_scale_leaky_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope);"
)

# Compile the fused CUDA kernel
fused_gemm = load_inline(
    name="fused_gemm",
    cuda_sources=fused_gemm_source,
    cpp_sources=fused_gemm_cpp_source,
    functions=["fused_gemm_scale_leaky_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize weights and bias like PyTorch's Linear layer
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.fused_gemm = fused_gemm  # Loaded CUDA module

    def forward(self, x):
        return self.fused_gemm.fused_gemm_scale_leaky_cuda(
            x,
            self.weight,
            self.bias,
            self.multiplier,
            self.negative_slope,
        )

# Ensure the same initialization inputs as the original model
batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]