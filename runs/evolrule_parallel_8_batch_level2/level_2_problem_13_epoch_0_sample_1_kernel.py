import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Mean Pool + Bias Add Kernel
mean_pool_bias_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_mean_pool_bias_add_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> bias,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int B, int C, int D, int H, int W) {

    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;

    if (b >= B || c >= C || h >= H || w >= W) return;

    scalar_t sum = 0;
    for (int d = 0; d < D; ++d) {
        sum += input[b][c][d][h][w];
    }
    output[b][c][0][h][w] = sum / D + bias[c][0][h][w];
}

torch::Tensor fused_mean_pool_bias_add_cuda(
    torch::Tensor input,
    torch::Tensor bias) {
    // input: (B, C, D, H, W)
    // bias: (1, C, 1, 1, 1) -> broadcasted to (B, C, 1, H, W)
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty({B, C, 1, H, W}, input.options());

    dim3 threads(32, 8); // W and H
    dim3 blocks(B, C);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_mean_pool_bias_add_cuda", ([&] {
        fused_mean_pool_bias_add_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            bias.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            B, C, D, H, W);
    }));

    return output;
}
"""

# Fused Softmax + Tanh + Scaling Kernel
softmax_tanh_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_softmax_tanh_scale_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    torch::PackedTensorAccessor<scalar_t,5> output,
    scalar_t scaling_factor,
    int B, int C, int H, int W) {

    int b = blockIdx.x;
    int h = threadIdx.y;
    int w = threadIdx.x;

    if (b >= B || h >= H || w >= W) return;

    // Compute softmax over channels
    scalar_t max_val = -INFINITY;
    for (int c = 0; c < C; ++c) {
        scalar_t val = input[b][c][0][h][w];
        if (val > max_val) max_val = val;
    }

    scalar_t sum_exp = 0;
    for (int c = 0; c < C; ++c) {
        sum_exp += exp(input[b][c][0][h][w] - max_val);
    }

    scalar_t denominator = sum_exp;

    for (int c = 0; c < C; ++c) {
        scalar_t softmax_val = exp(input[b][c][0][h][w] - max_val) / denominator;
        output[b][c][0][h][w] = tanh(softmax_val) * scaling_factor;
    }
}

torch::Tensor fused_softmax_tanh_scale_cuda(
    torch::Tensor input,
    scalar_t scaling_factor) {
    // input: (B, C, 1, H, W)
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty_like(input);

    dim3 threads(32, 8); // W and H
    dim3 blocks(B, 1); // blockIdx.y is unused

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_softmax_tanh_scale_cuda", ([&] {
        fused_softmax_tanh_scale_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            scaling_factor,
            B, C, H, W);
    }));

    return output;
}
"""

# Compile the CUDA kernels
mean_pool_bias_add = load_inline(
    name="mean_pool_bias_add",
    cuda_sources=mean_pool_bias_add_source,
    functions=["fused_mean_pool_bias_add_cuda"],
    verbose=True
)

softmax_tanh_scale = load_inline(
    name="softmax_tanh_scale",
    cuda_sources=softmax_tanh_scale_source,
    functions=["fused_softmax_tanh_scale_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.fused_mean_pool_bias_add = mean_pool_bias_add
        self.fused_softmax_tanh_scale = softmax_tanh_scale

    def forward(self, x):
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        # Apply fused mean pool + bias add
        x = self.fused_mean_pool_bias_add.fused_mean_pool_bias_add_cuda(x, self.bias)
        # Apply fused softmax, tanh, scaling
        x = self.fused_softmax_tanh_scale.fused_softmax_tanh_scale_cuda(x, self.scaling_factor)
        return x

# === Test config ===
batch_size = 16
in_channels  = 16  
out_channels = 64  
depth = 32; height = width = 128  
kernel_size  = 3
stride       = 1  
padding = 1
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]