import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size, int in_channels, int height_in, int width_in,
    int out_channels, int kernel_h, int kernel_w, int stride,
    int padding, int output_padding, int groups) {

    int height_out = (height_in - 1)*stride - 2*padding + kernel_h + output_padding;
    int width_out = (width_in - 1)*stride - 2*padding + kernel_w + output_padding;

    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * out_channels * height_out * width_out) return;

    int w_out = output_idx % width_out;
    int h_out = (output_idx / width_out) % height_out;
    int c_out = (output_idx / (width_out * height_out)) % out_channels;
    int n = output_idx / (width_out * height_out * out_channels);

    float val = 0.0;

    int out_channels_per_group = out_channels / groups;
    int group = c_out / out_channels_per_group;
    int in_channels_per_group = in_channels / groups;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = (h_out - kh + padding) / stride;
            int w_in = (w_out - kw + padding) / stride;

            if (h_in >= 0 && h_in < height_in && w_in >=0 && w_in < width_in) {
                for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
                    int c_in_full = group * in_channels_per_group + c_in;

                    int weight_offset = c_out * in_channels_per_group * kernel_h * kernel_w 
                                       + c_in * kernel_h * kernel_w 
                                       + kh * kernel_w + kw;

                    int input_offset = n * in_channels * height_in * width_in 
                                      + c_in_full * height_in * width_in 
                                      + h_in * width_in + w_in;

                    val += weight[weight_offset] * input[input_offset];
                }
            }
        }
    }

    int output_offset = n * out_channels * height_out * width_out 
                        + c_out * height_out * width_out 
                        + h_out * width_out + w_out;
    output[output_offset] = val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int kernel_size,
    int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);

    const int out_channels = weight.size(0);
    const int kernel_h = kernel_size;
    const int kernel_w = kernel_size;

    const int height_out = (height_in - 1)*stride - 2*padding + kernel_h + output_padding;
    const int width_out = (width_in - 1)*stride - 2*padding + kernel_w + output_padding;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, height_out, width_out}, options);

    const int threads = 256;
    const int elements = batch_size * out_channels * height_out * width_out;
    const int blocks = (elements + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, height_in, width_in,
        out_channels, kernel_h, kernel_w, stride,
        padding, output_padding, groups
    );

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int kernel_size, int groups);"
)

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-g", "-Wno-deprecated-gpu-targets"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Bind CUDA function
        self.conv_transpose2d = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding, self.kernel_size, self.groups
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output