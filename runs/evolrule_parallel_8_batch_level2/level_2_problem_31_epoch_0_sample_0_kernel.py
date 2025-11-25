import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* x,
    float* out,
    const float constant_value,
    const float* bias,
    float scaling_factor,
    int C,
    int H,
    int W,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        val = min(val, constant_value);
        int c = (idx / (H * W)) % C;
        val += bias[c];
        val *= scaling_factor;
        out[idx] = val;
    }
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor x,
    float constant_value,
    torch::Tensor bias,
    float scaling_factor
) {
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    auto bias_flat = bias.view({-1});

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        constant_value,
        bias_flat.data_ptr<float>(),
        scaling_factor,
        C,
        H,
        W,
        size
    );

    return out;
}
"""

fused_elementwise_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor x, float constant_value, torch::Tensor bias, float scaling_factor);"
)

# Load the fused kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp_source,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_elementwise = fused_elementwise  # Bind the fused kernel

    def forward(self, x):
        x = self.conv(x)
        # Apply the fused operations using the CUDA kernel
        x = self.fused_elementwise.fused_elementwise_cuda(
            x,
            self.constant_value,
            self.bias,
            self.scaling_factor,
        )
        return x

# The get_inputs and get_init_inputs functions remain unchanged
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