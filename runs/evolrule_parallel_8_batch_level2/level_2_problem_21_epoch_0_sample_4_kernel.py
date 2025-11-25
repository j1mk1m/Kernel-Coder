import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for bias addition, scaling, and sigmoid
fused_ops_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void fused_bias_scale_sigmoid_kernel(
    const float* x_data,
    const float* bias_data,
    const float* scale_data,
    float* out_data,
    int batch_size,
    int out_channels,
    int height,
    int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width)
        return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int n = idx / (out_channels * height * width);

    float bias_val = bias_data[c];
    float scale_val = scale_data[c];
    float x_val = x_data[idx];
    float temp = (x_val + bias_val) * scale_val;
    out_data[idx] = 1.0f / (1.0f + expf(-temp));
}

torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor scale) {
    auto batch_size = x.size(0);
    auto out_channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    auto out = torch::empty_like(x);

    const int threads = 256;
    int elements = x.numel();
    int blocks = (elements + threads - 1) / threads;

    fused_bias_scale_sigmoid_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, out_channels, height, width);

    return out;
}
"""

# Header for the fused operations
fused_ops_h = """
torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor x, torch::Tensor bias, torch::Tensor scale);
"""

# Compile the fused CUDA operations
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_h,
    cuda_sources=fused_ops_source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_ops = fused_ops  # Holds the CUDA function

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x