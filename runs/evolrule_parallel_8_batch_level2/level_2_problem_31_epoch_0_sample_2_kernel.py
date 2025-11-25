import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA code for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_ops_kernel(
    const float* x, 
    const float* bias,
    float* out,
    float constant_val,
    float scaling_factor,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    float val = x[idx];
    val = fminf(val, constant_val);
    val += bias[c];
    val *= scaling_factor;

    out[idx] = val;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float constant_val,
    float scaling_factor
) {
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int height = x.size(2);
    const int width = x.size(3);

    auto out = torch::empty_like(x);

    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        constant_val,
        scaling_factor,
        batch_size,
        channels,
        height,
        width
    );

    cudaDeviceSynchronize();
    return out;
}
"""

fused_ops_header = """
torch::Tensor fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float constant_val,
    float scaling_factor
);
"""

# Compile the fused operations CUDA code
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_ops_header,
    cuda_sources=fused_ops_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.constant_value, self.scaling_factor)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]