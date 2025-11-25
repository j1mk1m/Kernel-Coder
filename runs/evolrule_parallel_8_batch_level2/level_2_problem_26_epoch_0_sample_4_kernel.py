import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for element-wise addition and HardSwish
fused_add_hswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hswish_kernel(
    const float* x, const float* add_input, float* out, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = x[idx] + add_input[idx];
        float temp = sum + 3.0f;
        float relu6_val = fmaxf(0.0f, fminf(temp, 6.0f));
        float hswish_part = sum * relu6_val / 6.0f;
        out[idx] = sum * hswish_part;
    }
}

torch::Tensor fused_add_hswish_cuda(torch::Tensor x, torch::Tensor add_input) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hswish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), add_input.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}
"""

fused_add_hswish_cpp_source = "torch::Tensor fused_add_hswish_cuda(torch::Tensor x, torch::Tensor add_input);"

# Compile the fused CUDA kernel
fused_add_hswish = load_inline(
    name="fused_add_hswish",
    cpp_sources=fused_add_hswish_cpp_source,
    cuda_sources=fused_add_hswish_source,
    functions=["fused_add_hswish_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_hswish = fused_add_hswish  # Store the loaded module

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = self.fused_add_hswish.fused_add_hswish_cuda(x, add_input)
        return x

# Configuration parameters
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W).cuda(),
        torch.rand(batch_size, out_channels, D * stride, H * stride, W * stride).cuda()
    ]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]