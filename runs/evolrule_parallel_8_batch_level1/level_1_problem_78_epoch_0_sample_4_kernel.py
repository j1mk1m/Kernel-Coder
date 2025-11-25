import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv2d_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int out_h, int out_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_h * out_w)
        return;

    int x_out = idx % out_w;
    int y_out = (idx / out_w) % out_h;
    int c_out = (idx / (out_w * out_h)) % out_channels;
    int n = idx / (out_channels * out_h * out_w);

    float acc = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int y_in = (y_out + padding_h - ky) / stride_h;
                int x_in = (x_out + padding_w - kx) / stride_w;

                if (y_in >= 0 && y_in < in_h && x_in >= 0 && x_in < in_w) {
                    int w_offset = c_in * out_channels * kernel_h * kernel_w +
                                   c_out * kernel_h * kernel_w +
                                   ky * kernel_w + kx;
                    float w = weight[w_offset];

                    int in_offset = n * in_channels * in_h * in_w +
                                    c_in * in_h * in_w +
                                    y_in * in_w + x_in;
                    float in_val = input[in_offset];

                    acc += in_val * w;
                }
            }
        }
    }

    int out_offset = n * out_channels * out_h * out_w +
                     c_out * out_h * out_w +
                     y_out * out_w + x_out;
    output[out_offset] = acc;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = weight.size(1);

    int out_h = (in_h - 1) * stride_h - 2 * padding_h + kernel_h;
    int out_w = (in_w - 1) * stride_w - 2 * padding_w + kernel_w;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    int total_elements = batch_size * out_channels * out_h * out_w;
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    transposed_conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_h, in_w, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w,
        out_h, out_w
    );

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w);"
)

transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

transposed_conv_cuda = transposed_conv.transposed_conv2d_cuda

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output = transposed_conv_cuda(x, self.weight, stride_h, stride_w, padding_h, padding_w)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output