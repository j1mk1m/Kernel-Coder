import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_ops_kernel(
    const float* x,
    const float* scaling_factor,
    const float* bias,
    float* out,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) {
        return;
    }

    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % out_channels;
    int n = idx / (out_channels * depth * height * width);

    float scale_val = scaling_factor[c * 1 * 1 * 1];
    float bias_val = bias[c * 1 * 1 * 1];

    float value = x[idx];
    float temp = value * scale_val;
    temp = tanhf(temp);
    temp *= bias_val;
    float sigmoid_val = 1.0f / (1.0f + expf(-temp));
    out[idx] = sigmoid_val;
}

torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias) {
    const int batch_size = x.size(0);
    const int out_channels = x.size(1);
    const int depth = x.size(2);
    const int height = x.size(3);
    const int width = x.size(4);

    auto out = torch::empty_like(x);

    const int num_elements = batch_size * out_channels * depth * height * width;
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_ops_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_channels,
        depth,
        height,
        width
    );

    return out;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias);"
)

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x, self.scaling_factor, self.bias)
        return x