import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation CUDA kernel
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activation_kernel(const float* input, float add_val, float scale, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float s;
        if (x > 0) {
            float exp_neg_x = expf(-x);
            s = x + log1pf(exp_neg_x);
        } else {
            float exp_x = expf(x);
            s = log1pf(exp_x);
        }
        float tanh_s = tanhf(s);
        float mish = x * tanh_s;
        mish += add_val;
        if (mish < -1.0f) {
            mish = -1.0f;
        } else if (mish > 1.0f) {
            mish = 1.0f;
        }
        mish *= scale;
        output[idx] = mish;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input, float add_val, float scale) {
    const int size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), add_val, scale, output.data_ptr<float>(), size
    );

    return output;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor input, float add_val, float scale);"
)

# Compile the fused activation kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_activation = fused_activation  # The loaded CUDA module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_activation.fused_activation_cuda(x, self.add_value, self.scale)
        return x

# The get_inputs and get_init_inputs functions remain the same as in the original code
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]