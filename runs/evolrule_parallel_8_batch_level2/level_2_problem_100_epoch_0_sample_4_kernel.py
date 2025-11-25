import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused clamp and division CUDA kernel
fused_clamp_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_clamp_divide_kernel(
    const float* input, 
    float* output, 
    float min_val, 
    float divisor, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(input[idx], min_val) / divisor;
    }
}

torch::Tensor fused_clamp_divide_cuda(
    torch::Tensor input, 
    float min_val, 
    float divisor
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_clamp_divide_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        min_val,
        divisor,
        size
    );

    return output;
}
"""

fused_clamp_divide_header = (
    "torch::Tensor fused_clamp_divide_cuda(torch::Tensor input, float min_val, float divisor);"
)

# Compile the fused kernel
fused_clamp_divide = load_inline(
    name="fused_clamp_divide",
    cpp_sources=fused_clamp_divide_header,
    cuda_sources=fused_clamp_divide_source,
    functions=["fused_clamp_divide_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        min_value, 
        divisor
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.min_value = min_value
        self.divisor = divisor
        self.fused_clamp_divide = fused_clamp_divide

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_clamp_divide.fused_clamp_divide_cuda(
            x, self.min_value, self.divisor
        )
        return x

# The following functions are reused from the original code
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]