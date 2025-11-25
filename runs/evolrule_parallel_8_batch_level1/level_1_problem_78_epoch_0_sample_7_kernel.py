import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel code
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ kernel,
    scalar_t* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int H_out, int W_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N * C_out * H_out * W_out) return;

    int out_w = idx % W_out;
    int rem = idx / W_out;
    int out_h = rem % H_out;
    rem = rem / H_out;
    int out_c = rem % C_out;
    int n = rem / C_out;

    scalar_t acc = 0.0;

    for (int in_c = 0; in_c < C_in; ++in_c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = (out_h - kh + 2 * padding_h) / stride_h;
                int w = (out_w - kw + 2 * padding_w) / stride_w;

                if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                    int input_offset = n * C_in * H_in * W_in
                                      + in_c * H_in * W_in
                                      + h * W_in + w;

                    int kernel_offset = out_c * C_in * kernel_h * kernel_w
                                       + in_c * kernel_h * kernel_w
                                       + kh * kernel_w + kw;

                    acc += input[input_offset] * kernel[kernel_offset];
                }
            }
        }
    }

    int output_offset = n * C_out * H_out * W_out
                       + out_c * H_out * W_out
                       + out_h * W_out + out_w;

    output[output_offset] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int H_out, int W_out
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (output.numel() + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C_in, H_in, W_in,
            C_out, kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            H_out, W_out);
    }));

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor kernel, torch::Tensor output, int N, int C_in, int H_in, int W_in, int C_out, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int H_out, int W_out);"
)

# Compile the CUDA kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias  # The original model has bias=False, so this is for flexibility

        # Initialize weight parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.register_parameter('bias_param', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H_in, W_in = x.size()
        C_out = self.out_channels
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        # Compute output dimensions
        H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h
        W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_w

        output = torch.empty(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)

        # Launch CUDA kernel
        output = conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            output,
            N, C_in, H_in, W_in,
            C_out, kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            H_out, W_out
        )

        # Add bias if present (original model has bias=False, so this is optional)
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1)

        return output