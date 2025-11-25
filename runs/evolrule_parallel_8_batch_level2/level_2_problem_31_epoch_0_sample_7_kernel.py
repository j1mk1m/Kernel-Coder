import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for min, add bias, and scaling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>  // for std::min

__global__ void fused_ops_kernel(
    const float* in_data,
    const float* bias_data,
    float constant,
    float scaling,
    float* out_data,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (C * W * H);

    float val = in_data[idx];
    val = std::min(val, constant);
    val += bias_data[c];  // bias is [C,1,1], so bias[c] is correct
    val *= scaling;
    out_data[idx] = val;
}

torch::Tensor fused_ops_cuda(
    torch::Tensor in,
    torch::Tensor bias,
    float constant,
    float scaling
) {
    auto output = torch::empty_like(in);
    const int N = in.size(0);
    const int C = in.size(1);
    const int H = in.size(2);
    const int W = in.size(3);
    const int num_elements = N * C * H * W;

    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    fused_ops_kernel<<<grid_size, block_size>>>(
        in.data_ptr<float>(),
        bias.data_ptr<float>(),
        constant,
        scaling,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_ops_cuda(torch::Tensor in, torch::Tensor bias, float constant, float scaling);"
)

# Load the CUDA extension
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
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape).cuda())  # Initialize on CUDA
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops  # The loaded CUDA module

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