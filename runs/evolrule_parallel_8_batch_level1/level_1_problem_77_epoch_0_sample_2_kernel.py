import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename scalar_t>
__global__ void conv_transpose3d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int depth_out,
    int height_out,
    int width_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * depth_out * height_out * width_out) {
        return;
    }

    // Compute output indices
    int w_out = index % width_out;
    int h_out = (index / width_out) % height_out;
    int d_out = (index / (width_out * height_out)) % depth_out;
    int c_out = (index / (width_out * height_out * depth_out)) % out_channels;
    int n = index / (width_out * height_out * depth_out * out_channels);

    float acc = 0.0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size_d; ++kd) {
            for (int kh = 0; kh < kernel_size_h; ++kh) {
                for (int kw = 0; kw < kernel_size_w; ++kw) {
                    int d_in = (d_out - (kd * dilation_d - padding_d)) / stride_d;
                    int h_in = (h_out - (kh * dilation_h - padding_h)) / stride_h;
                    int w_in = (w_out - (kw * dilation_w - padding_w)) / stride_w;

                    if (d_in < 0 || d_in >= depth_in) continue;
                    if (h_in < 0 || h_in >= height_in) continue;
                    if (w_in < 0 || w_in >= width_in) continue;

                    int input_offset = n * in_channels * depth_in * height_in * width_in +
                        c_in * depth_in * height_in * width_in +
                        d_in * height_in * width_in +
                        h_in * width_in +
                        w_in;

                    int weight_offset = c_in * out_channels * kernel_size_d * kernel_size_h * kernel_size_w +
                        c_out * kernel_size_d * kernel_size_h * kernel_size_w +
                        kd * kernel_size_h * kernel_size_w +
                        kh * kernel_size_w +
                        kw;

                    acc += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[c_out];
    }

    int output_offset = n * out_channels * depth_out * height_out * width_out +
        c_out * depth_out * height_out * width_out +
        d_out * height_out * width_out +
        h_out * width_out +
        w_out;

    output[output_offset] = acc;
}

torch::Tensor conv_transpose3d_forward_cuda(torch::Tensor input, 
                                           torch::Tensor weight,
                                           torch::Tensor bias,
                                           int stride_d, int stride_h, int stride_w,
                                           int padding_d, int padding_h, int padding_w,
                                           int dilation_d, int dilation_h, int dilation_w) {
    bool has_bias = (bias.defined() && bias.sizes().size() > 0 && bias.size(0) == weight.size(1));

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int kernel_size_d = weight.size(2);
    int kernel_size_h = weight.size(3);
    int kernel_size_w = weight.size(4);

    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + 1;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + 1;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + 1;

    torch::Tensor output = torch::zeros({batch_size, weight.size(1), depth_out, height_out, width_out}, input.options());

    int num_elements = output.numel();

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward_cuda", ([&] {
        conv_transpose3d_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            has_bias ? bias.data<scalar_t>() : nullptr,
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            weight.size(1),
            depth_in,
            height_in,
            width_in,
            kernel_size_d,
            kernel_size_h,
            kernel_size_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
            depth_out,
            height_out,
            width_out
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp_header = """
#include <torch/extension.h>
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_header,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_d = self.stride
        stride_h = self.stride
        stride_w = self.stride
        padding_d = self.padding
        padding_h = self.padding
        padding_w = self.padding
        dilation_d = self.dilation
        dilation_h = self.dilation
        dilation_w = self.dilation

        bias = self.bias_param if self.bias_param is not None else torch.empty(0, dtype=x.dtype, device=x.device)

        return conv_transpose3d.conv_transpose3d_forward_cuda(
            x,
            self.weight,
            bias,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w
        )