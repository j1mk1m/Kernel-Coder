import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused operation CUDA kernel
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_leaky_relu_mult_leaky_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ multiplier,
    scalar_t* __restrict__ output,
    int N, int C, int D, int H, int W) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N*C*D*H*W) return;

    int w = index % W;
    int h = (index / W) % H;
    int d = (index / (W*H)) % D;
    int c = (index / (W*H*D)) % C;
    int n = index / (C*W*H*D);

    scalar_t val = input[index];

    // First LeakyReLU
    val = val > 0 ? val : val * 0.2;

    // Multiply by multiplier
    val *= multiplier[c];

    // Second LeakyReLU
    val = val > 0 ? val : val * 0.2;

    output[index] = val;
}

torch::Tensor fused_leaky_relu_mult_leaky_cuda(
    torch::Tensor input,
    torch::Tensor multiplier) {

    // Ensure multiplier has shape [C]
    int C = input.size(1);
    assert(multiplier.size(0) == C);

    int N = input.size(0);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int elements = N * C * D * H * W;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_leaky_relu_mult_leaky_cuda", ([&]{
        fused_leaky_relu_mult_leaky_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            multiplier.data<scalar_t>(),
            output.data<scalar_t>(),
            N, C, D, H, W
        );
    }));

    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_leaky_relu_mult_leaky_cuda(torch::Tensor input, torch::Tensor multiplier);"
)

# Compile the fused operation
fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_leaky_relu_mult_leaky_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.fused_op = fused_op  # Bind the fused operation

    def forward(self, x):
        x = self.conv_transpose(x)
        # Reshape multiplier to 1D for the kernel
        x = self.fused_op.fused_leaky_relu_mult_leaky_cuda(x, self.multiplier.view(-1))
        x = self.max_pool(x)
        return x

# Define global variables as in the original code
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]