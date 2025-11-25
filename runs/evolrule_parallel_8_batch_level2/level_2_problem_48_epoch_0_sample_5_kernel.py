import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused kernel for element-wise operations after convolution
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_operations_kernel(
    const scalar_t* __restrict__ conv_out,
    const scalar_t* __restrict__ scaling_factor,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    // Compute indices
    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % out_channels;
    int b = idx / (out_channels * depth * height * width);

    // Apply scaling factor (element-wise multiplication)
    scalar_t scaled = conv_out[idx] * scaling_factor[c * depth * height * width + 0]; // since scaling_factor is (out_channels, 1, 1, 1)

    // Apply tanh
    scaled = tanh(scaled);

    // Apply bias multiplication
    scaled *= bias[c * depth * height * width + 0]; // same logic for bias shape

    // Apply sigmoid
    output[idx] = 1.0 / (1.0 + exp(-scaled));
}

torch::Tensor fused_operations_cuda(
    torch::Tensor conv_out,
    torch::Tensor scaling_factor,
    torch::Tensor bias
) {
    auto batch_size = conv_out.size(0);
    auto out_channels = conv_out.size(1);
    auto depth = conv_out.size(2);
    auto height = conv_out.size(3);
    auto width = conv_out.size(4);

    auto output = torch::empty_like(conv_out);

    const int threads = 256;
    int elements = batch_size * out_channels * depth * height * width;
    int blocks = (elements + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(conv_out.type(), "fused_operations_cuda", ([&] {
        fused_operations_kernel<scalar_t><<<blocks, threads>>>(
            conv_out.data_ptr<scalar_t>(),
            scaling_factor.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, out_channels, depth, height, width
        );
    }));

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_operations_cuda(torch::Tensor, torch::Tensor, torch::Tensor);"
)

# Compile the fused operations kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops  # Load the fused kernel

    def forward(self, x):
        x = self.conv(x)
        # Apply all fused operations in a single kernel call
        x = self.fused_ops.fused_operations_cuda(x, self.scaling_factor, self.bias)
        return x

def get_inputs():
    return [torch.rand(128, 3, 16, 64, 64).cuda()]

def get_init_inputs():
    return [128, 3, 16, (16, 1, 1, 1)]