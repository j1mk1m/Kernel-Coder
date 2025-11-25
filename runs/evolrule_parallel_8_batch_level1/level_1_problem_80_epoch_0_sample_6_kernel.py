import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int H, int W,
    int kernel_h, int kernel_w,
    int stride,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int height_out, int width_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height_out * width_out) {
        return;
    }

    int w_out = idx % width_out;
    int tmp = idx / width_out;
    int h_out = tmp % height_out;
    tmp /= height_out;
    int c_out = tmp % out_channels;
    int n = tmp / out_channels;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_out * stride + kh * dilation_h - pad_h;
            int w_in = w_out * stride + kw * dilation_w - pad_w;

            if (h_in < pad_h || h_in >= (pad_h + H) || 
                w_in < pad_w || w_in >= (pad_w + W)) {
                continue;
            }

            h_in -= pad_h;
            w_in -= pad_w;

            for (int c_in = 0; c_in < in_channels; ++c_in) {
                int input_offset = n * in_channels * H * W +
                    c_in * H * W +
                    h_in * W + w_in;
                float input_val = input[input_offset];

                int weight_offset = c_out * in_channels * kernel_h * kernel_w +
                    c_in * kernel_h * kernel_w +
                    kh * kernel_w + kw;
                float weight_val = weight[weight_offset];

                sum += input_val * weight_val;
            }
        }
    }

    int output_offset = n * out_channels * height_out * width_out +
                        c_out * height_out * width_out +
                        h_out * width_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int in_channels,
    int out_channels,
    std::tuple<int, int> kernel_size,
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation
) {
    int kernel_h = std::get<0>(kernel_size);
    int kernel_w = std::get<1>(kernel_size);
    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int B = input.size(0);
    int H = input.size(2);
    int W = input.size(3);

    int height_out = (H + 2*pad_h - dilation_h*(kernel_h -1) -1)/stride +1;
    int width_out = (W + 2*pad_w - dilation_w*(kernel_w -1) -1)/stride +1;

    auto output = torch::empty({B, out_channels, height_out, width_out}, input.options());

    int threads_per_block = 256;
    int blocks_per_grid = (B * out_channels * height_out * width_out + threads_per_block -1) / threads_per_block;

    conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, in_channels, out_channels,
        H, W,
        kernel_h, kernel_w,
        stride, pad_h, pad_w,
        dilation_h, dilation_w,
        height_out, width_out
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int in_channels,
    int out_channels,
    std::tuple<int, int> kernel_size,
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation
);
"""

conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return conv2d.conv2d_cuda(
            x, self.weight,
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride,
            self.padding, self.dilation
        )