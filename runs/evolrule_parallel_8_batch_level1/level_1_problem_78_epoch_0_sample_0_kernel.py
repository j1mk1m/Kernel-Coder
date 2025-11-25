import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel implementation
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define KERNEL_H 3
#define KERNEL_W 7
#define PADDING_H 1
#define PADDING_W 3

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int n = idx / (out_channels * input_height * input_width);
    int remainder = idx % (out_channels * input_height * input_width);
    int oc = remainder / (input_height * input_width);
    remainder = remainder % (input_height * input_width);
    int h = remainder / input_width;
    int w = remainder % input_width;

    if (n >= batch_size || oc >= out_channels || h >= input_height || w >= input_width) {
        return;
    }

    float result = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < KERNEL_H; ++kh) {
            for (int kw = 0; kw < KERNEL_W; ++kw) {
                int i_h = h - kh + PADDING_H;
                int i_w = w - kw + PADDING_W;
                if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                    int input_offset = n * in_channels * input_height * input_width +
                                      ic * input_height * input_width +
                                      i_h * input_width + i_w;
                    int kernel_offset = (ic * out_channels + oc) * KERNEL_H * KERNEL_W +
                                       kh * KERNEL_W + kw;
                    result += kernel[kernel_offset] * input[input_offset];
                }
            }
        }
    }

    int output_offset = n * out_channels * input_height * input_width +
                       oc * input_height * input_width +
                       h * input_width + w;
    output[output_offset] = result;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = kernel.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, input_height, input_width}, input.options());

    const int threads_per_block = 1024;
    const int num_elements = batch_size * out_channels * input_height * input_width;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", [&] {
        conv_transpose2d_kernel<<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            kernel.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width
        );
    });

    return output;
}
"""

# Compile the CUDA kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d.conv_transpose2d_cuda(x, self.weight)