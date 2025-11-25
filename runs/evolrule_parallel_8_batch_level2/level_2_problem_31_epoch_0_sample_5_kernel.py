import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for min, add bias, and scaling
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_min_add_scale_kernel(
    const float* conv_out, const float* bias, float* out,
    const float constant_val, const float scaling_factor, int batch, int channels, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * height * width)
        return;

    // Compute the channel index to handle bias broadcasting
    int channel = (idx / (height * width)) % channels;

    // Apply min with constant, add bias (broadcasted), then scale
    float val = conv_out[idx];
    val = min(val, constant_val);
    val += bias[channel];  // bias is (channels, 1, 1), so only depends on channel
    val *= scaling_factor;

    out[idx] = val;
}

torch::Tensor fused_min_add_scale_cuda(
    torch::Tensor conv_out, torch::Tensor bias,
    float constant_val, float scaling_factor) {

    const int batch = conv_out.size(0);
    const int channels = conv_out.size(1);
    const int height = conv_out.size(2);
    const int width = conv_out.size(3);
    const int size = batch * channels * height * width;

    auto out = torch::empty_like(conv_out);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_min_add_scale_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        constant_val, scaling_factor, batch, channels, height, width);

    return out;
}
"""

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources="torch::Tensor fused_min_add_scale_cuda(torch::Tensor, torch::Tensor, float, float);",
    cuda_sources=fused_ops_source,
    functions=["fused_min_add_scale_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops  # Access the compiled kernel

    def forward(self, x):
        x = self.conv(x)
        # Use the fused kernel for min, bias add, and scaling
        return self.fused_ops.fused_min_add_scale_cuda(
            x, self.bias, self.constant_value, self.scaling_factor
        )

# Ensure get_inputs and get_init_inputs remain unchanged as per original
def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]