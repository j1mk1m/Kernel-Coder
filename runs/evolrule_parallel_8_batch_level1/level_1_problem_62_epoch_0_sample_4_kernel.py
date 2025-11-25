import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <bool HAS_BIAS>
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int output_height,
    int output_width,
    int stride,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (out_channels * output_height * output_width);

    float sum = 0.0;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int h_in = h_out * stride + kh * dilation_h - padding_h;
                int w_in = w_out * stride + kw * dilation_w - padding_w;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    const int input_offset = n * in_channels * input_height * input_width +
                                            c_in * input_height * input_width +
                                            h_in * input_width + w_in;
                    const int weight_offset = c_out * in_channels * kernel_height * kernel_width +
                                             c_in * kernel_height * kernel_width +
                                             kh * kernel_width + kw;
                    sum += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    if constexpr (HAS_BIAS) {
        sum += bias[c_out];
    }

    const int output_offset = n * out_channels * output_height * output_width +
                             c_out * output_height * output_width +
                             h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         int stride,
                         int padding_h, int padding_w,
                         int dilation_h, int dilation_w,
                         int kernel_height, int kernel_width) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * output_height * output_width;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    if (bias.defined()) {
        conv2d_kernel<true><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels, input_height, input_width,
            kernel_height, kernel_width,
            output_height, output_width,
            stride, padding_h, padding_w, dilation_h, dilation_w
        );
    } else {
        conv2d_kernel<false><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels, input_height, input_width,
            kernel_height, kernel_width,
            output_height, output_width,
            stride, padding_h, padding_w, dilation_h, dilation_w
        );
    }

    cudaDeviceSynchronize();
    return output;
}
"""

conv2d_cpp_source = """
extern "C" {
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         int stride,
                         int padding_h, int padding_w,
                         int dilation_h, int dilation_w,
                         int kernel_height, int kernel_width);
}
"""

conv2d_cuda = load_inline(
    name="conv2d_cuda",
    cuda_sources=conv2d_cuda_source,
    cpp_sources=conv2d_cpp_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv2d
        weight = conv.weight
        bias = conv.bias if conv.bias is not None else None
        stride = conv.stride[0]
        padding_h, padding_w = conv.padding
        dilation_h, dilation_w = conv.dilation
        kernel_height, kernel_width = conv.kernel_size

        return conv2d_cuda(
            x,
            weight,
            bias,
            stride,
            padding_h, padding_w,
            dilation_h, dilation_w,
            kernel_height, kernel_width
        )