import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, int.__init__())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize convolution weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.bias_param = None

        # Load the custom CUDA kernel
        conv2d_cuda_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int stride,
    const int pad_h,
    const int pad_w,
    const int dil_h,
    const int dil_w,
    const int output_height,
    const int output_width,
    const scalar_t* __restrict__ bias) {

    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_x >= output_width) return;

    for (int c = threadIdx.x; c < out_channels; c += blockDim.x) {
        scalar_t sum = 0;
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int in_y = out_y * stride + ky * dil_h - pad_h;
                int in_x = out_x * stride + kx * dil_w - pad_w;
                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    for (int ch = 0; ch < in_channels; ++ch) {
                        sum += weight[c * in_channels * kernel_height * kernel_width + ch * kernel_height * kernel_width + ky * kernel_width + kx] *
                               input[batch_idx * in_channels * input_height * input_width + ch * input_height * input_width + in_y * input_width + in_x];
                    }
                }
            }
        }
        if (bias) sum += bias[c];
        output[batch_idx * out_channels * output_height * output_width + c * output_height * output_width + out_y * output_width + out_x] = sum;
    }
}

torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, std::tuple<int, int> padding, std::tuple<int, int> dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_height = weight.size(2);
    const int kernel_width = weight.size(3);
    const int pad_h = std::get<0>(padding);
    const int pad_w = std::get<1>(padding);
    const int dil_h = std::get<0>(dilation);
    const int dil_w = std::get<1>(dilation);
    const int output_height = (input_height + 2 * pad_h - dil_h * (kernel_height - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * pad_w - dil_w * (kernel_width - 1) - 1) / stride + 1;
    const int out_channels = weight.size(0);

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(256);
    dim3 blocks(batch_size, output_height, (output_width + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv2d_cuda", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            stride,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            output_height,
            output_width,
            bias.data_ptr<scalar_t>());
    }));

    return output;
}
"""

        self.conv2d_cuda = load_inline(
            name="conv2d_cuda",
            cpp_sources="""
            torch::Tensor custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, std::tuple<int, int> padding, std::tuple<int, int> dilation);
            """,
            cuda_sources=conv2d_cuda_source,
            functions=["custom_conv2d"],
            verbose=True
        )

    def forward(self, x):
        # Calculate padding
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        # Pad input tensor
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))

        # Launch custom kernel
        output = self.conv2d_cuda.custom_conv2d(
            x,
            self.weight,
            self.bias_param if self.bias else torch.empty(0, device=x.device),
            self.stride,
            self.padding,
            self.dilation
        )

        return output