import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA code for fused operations (add bias, scale, sigmoid)
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel(
    const float* input,
    const float* bias,
    float scaling_factor,
    float* output,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W)
        return;

    // Compute the channel index
    int c = (idx / (H * W)) % C;

    float val = input[idx];
    val += bias[c]; // bias is (C,1,1)
    val *= scaling_factor;
    val = 1.0f / (1.0f + expf(-val)); // Sigmoid
    output[idx] = val;
}

torch::Tensor fused_operations(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Check bias shape
    assert(bias.sizes() == torch::IntArrayRef({C, 1, 1}));

    auto output = torch::empty_like(input);
    const int block_size = 256;
    int total_elements = N * C * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_operations(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor
);
"""

# Load the fused operations into a module
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_operations"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_operations = fused_operations

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.softmax(x, dim=1)
        x = self.fused_operations.fused_operations(x, self.bias, self.scaling_factor)
        return x