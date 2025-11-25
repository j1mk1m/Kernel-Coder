import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Mish + add + Hardtanh + scale
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_mish_add_clamp_scale_kernel(
    const float* input, float* output,
    const float add_val, const float scale,
    int num_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        float x = input[tid];
        float exp_x = expf(x);
        float softplus = logf(1.0f + exp_x);
        float tanh_softplus = tanhf(softplus);
        float mish_val = x * tanh_softplus;

        mish_val += add_val;
        if (mish_val < -1.0f) mish_val = -1.0f;
        else if (mish_val > 1.0f) mish_val = 1.0f;
        mish_val *= scale;

        output[tid] = mish_val;
    }
}

torch::Tensor fused_mish_add_clamp_scale(
    torch::Tensor input,
    float add_val,
    float scale
) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();

    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    fused_mish_add_clamp_scale_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_val,
        scale,
        num_elements
    );

    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_mish_add_clamp_scale(torch::Tensor input, float add_val, float scale);"
)

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_mish_add_clamp_scale"],
    verbose=True,
    extra_cflags=["-O3", "-ffast-math"],
    extra_ldflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_mish_add_clamp_scale(x, self.add_value, self.scale)
        return x

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