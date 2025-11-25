import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused clamp and divide CUDA kernel
clamp_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_divide_kernel(
    const float* input,
    float* output,
    float min_value,
    float divisor,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = fmaxf(val, min_value);
        output[idx] = val / divisor;
    }
}

torch::Tensor clamp_divide_cuda(
    torch::Tensor input,
    float min_value,
    float divisor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_divide_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        min_value,
        divisor,
        size
    );

    return output;
}
"""

# Compile the fused kernel
clamp_divide_cpp_source = (
    "torch::Tensor clamp_divide_cuda(torch::Tensor input, float min_value, float divisor);"
)

clamp_divide = load_inline(
    name="clamp_divide",
    cpp_sources=clamp_divide_cpp_source,
    cuda_sources=clamp_divide_source,
    functions=["clamp_divide_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )
        self.min_value = min_value
        self.divisor = divisor
        self.clamp_divide = clamp_divide

    def forward(self, x):
        # Perform convolution
        conv_out = self.conv_transpose(x)
        # Apply fused clamp and divide
        result = self.clamp_divide.clamp_divide_cuda(
            conv_out, self.min_value, self.divisor
        )
        return result

# Ensure compatibility with get_init_inputs
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

# The original variables must be defined in the global scope for get_init_inputs
# Assuming these are defined as in the original code:
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