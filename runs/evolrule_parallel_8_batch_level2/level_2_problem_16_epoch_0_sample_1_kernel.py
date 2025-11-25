import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_mish_add_clamp_scale_kernel(
    const float* x_data,
    float add_val,
    float scale_val,
    float* out_data,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float x_val = x_data[idx];

    // Compute softplus
    float softplus_x;
    if (x_val >= 0.0f) {
        softplus_x = x_val + logf(1.0f + expf(-x_val));
    } else {
        softplus_x = logf(1.0f + expf(x_val));
    }

    // Compute tanh(softplus_x)
    float tanh_softplus = tanhf(softplus_x);

    // Mish activation
    float mish = x_val * tanh_softplus;

    // Add add_val
    mish += add_val;

    // Clamp between -1 and 1 (hardtanh)
    if (mish < -1.0f) {
        mish = -1.0f;
    } else if (mish > 1.0f) {
        mish = 1.0f;
    }

    // Multiply by scale
    mish *= scale_val;

    out_data[idx] = mish;
}

torch::Tensor fused_mish_add_clamp_scale_cuda(torch::Tensor x, float add_val, float scale_val) {
    const int num_elements = x.numel();
    auto out = torch::empty_like(x);

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_mish_add_clamp_scale_kernel<<<blocks_per_grid, threads_per_block>>>(
        x.data_ptr<float>(),
        add_val,
        scale_val,
        out.data_ptr<float>(),
        num_elements
    );

    return out;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_mish_add_clamp_scale_cuda(torch::Tensor x, float add_val, float scale_val);"
)

# Compile the fused kernel
fused_mish_add_clamp_scale = load_inline(
    name="fused_mish_add_clamp_scale",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_mish_add_clamp_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_mish_add_clamp_scale = fused_mish_add_clamp_scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_mish_add_clamp_scale.fused_mish_add_clamp_scale_cuda(
            x, self.add_value, self.scale
        )
        return x

# Global variables as in the original code
batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]