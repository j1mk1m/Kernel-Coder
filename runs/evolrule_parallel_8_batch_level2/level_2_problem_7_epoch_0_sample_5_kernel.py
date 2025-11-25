import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation CUDA kernel
fused_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_activations_kernel(
    const float* input,
    const float* bias,
    float* output,
    int total_size,
    int C,
    int D,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    float x = input[idx];

    // Apply ReLU
    x = fmaxf(0.0f, x);

    // Apply GELU approximation
    float inner = (M_SQRT2 / M_SQRTPI) * (x + 0.044715f * x * x * x);
    float tanh_inner = tanhf(inner);
    x = 0.5f * x * (1.0f + tanh_inner);

    // Apply Sigmoid
    x = 1.0f / (1.0f + expf(-x));

    // Compute channel index
    int c = (idx / (D * H * W)) % C;

    // Add bias
    x += bias[c];

    output[idx] = x;
}

torch::Tensor fused_activations_cuda(torch::Tensor input, torch::Tensor bias) {
    auto total_size = input.numel();
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;

    fused_activations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_size,
        C, D, H, W
    );

    return output;
}
"""

# Compile the CUDA code
fused_activations = load_inline(
    name="fused_activations",
    cpp_sources="",
    cuda_sources=fused_source,
    functions=["fused_activations_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activations_cuda = fused_activations.fused_activations_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activations_cuda(x, self.bias)
        return x

def get_inputs():
    batch_size = 64
    in_channels = 8
    depth, height, width = 32, 64, 64
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 8
    out_channels = 32
    kernel_size = 3
    bias_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, bias_shape]