import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM and GroupNorm
fused_gemm_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_groupnorm_kernel(const float* x, const float* weight, const float* bias, const float* running_mean, const float* running_var, float* out, int batch_size, int in_features, int out_features, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int i = idx / out_features;
    int o = idx % out_features;

    // Compute GEMM
    float gemm_result = 0.0f;
    for (int j = 0; j < in_features; ++j) {
        gemm_result += x[i * in_features + j] * weight[o * in_features + j];
    }

    // Apply GroupNorm
    float mean = 0.0f;
    float var = 0.0f;
    for (int j = 0; j < out_features; ++j) {
        mean += gemm_result;
        var += gemm_result * gemm_result;
    }
    mean /= out_features;
    var /= out_features;
    var -= mean * mean;
    var = max(var, eps);
    float inv_std = rsqrt(var);

    out[idx] = (gemm_result - mean) * inv_std * gamma[o] + beta[o];
}
"""

fused_gemm_groupnorm_cpp_source = (
    "void fused_gemm_groupnorm_cuda(const float* x, const float* weight, const float* bias, const float* running_mean, const float* running_var, float* out, int batch_size, int in_features, int out_features, float eps);"
)

# Compile the inline CUDA code for fused GEMM and GroupNorm
fused_gemm_groupnorm = load_inline(
    name="fused_gemm_groupnorm",
    cpp_sources=fused_gemm_groupnorm_cpp_source,
    cuda_sources=fused_gemm_groupnorm_source,
    functions=["fused_gemm_groupnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] *= sigmoid(x[idx]);
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void swish_cuda(float* x, int size) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x, size);
}
"""

swish_cpp_source = (
    "void swish_cuda(float* x, int size);"
)

# Compile the inline CUDA code for Swish activation
swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.fused_gemm_groupnorm = fused_gemm_groupnorm
        self.swish = swish
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.fused_gemm_groupnorm.fused_gemm_groupnorm_cuda(x, self.weight, self.bias, self.running_mean, self.running_var, x, x.size(0), x.size(1), x.size(2), 1e-5)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.swish.swish_cuda(x, x.size(0) * x.size(1))
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * self.multiply_weight
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.swish.swish_cuda(x, x.size(0) * x.size(1))
        return x

# Initialize the model with the provided parameters
batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

model_new = ModelNew(in_features, out_features, num_groups, multiply_weight_shape)

# Get the input tensor
inputs = get_inputs()

# Forward pass through the model
output = model_new(inputs[0])
print(output.shape)