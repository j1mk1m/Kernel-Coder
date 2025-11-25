import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for clamp and division
fused_clamp_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_clamp_div_kernel(
    const float* input,
    float* output,
    float min_val,
    float divisor_reciprocal,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx];
        val = fmaxf(val, min_val);
        output[idx] = val * divisor_reciprocal;
    }
}

torch::Tensor fused_clamp_div(
    torch::Tensor input,
    float min_val,
    float divisor_reciprocal
) {
    const int threads_per_block = 256;
    const int blocks = (input.numel() + threads_per_block - 1) / threads_per_block;

    auto output = torch::empty_like(input);

    fused_clamp_div_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        min_val,
        divisor_reciprocal,
        input.numel()
    );

    return output;
}
"""

fused_clamp_div_cpp_source = (
    "torch::Tensor fused_clamp_div(torch::Tensor input, float min_val, float divisor_reciprocal);"
)

# Compile the fused kernel
fused_clamp_div = load_inline(
    name="fused_clamp_div",
    cpp_sources=fused_clamp_div_cpp_source,
    cuda_sources=fused_clamp_div_source,
    functions=["fused_clamp_div"],
    verbose=True,
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
        self.divisor_reciprocal = 1.0 / divisor  # Precompute reciprocal for efficiency

    def forward(self, x):
        # Transposed convolution using PyTorch's optimized implementation
        x = self.conv_transpose(x)
        # Apply fused clamp and division
        return fused_clamp_div.fused_clamp_div(
            x,
            self.min_value,
            self.divisor_reciprocal
        )