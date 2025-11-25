import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w
) {
    CUDA_KERNEL_LOOP(index, batch_size * out_channels * depth_out * height_out * width_out) {
        int w_out = index % width_out;
        int h_out = (index / width_out) % height_out;
        int d_out = (index / (width_out * height_out)) % depth_out;
        int o_c = (index / (width_out * height_out * depth_out)) % out_channels;
        int b = index / (width_out * height_out * depth_out * out_channels);

        scalar_t sum = 0.0;

        for (int i_c = 0; i_c < in_channels; ++i_c) {
            for (int k_d = 0; k_d < kernel_depth; ++k_d) {
                for (int k_h = 0; k_h < kernel_height; ++k_h) {
                    for (int k_w = 0; k_w < kernel_width; ++k_w) {
                        int d_in = (d_out + padding_d - k_d) / stride_d;
                        int h_in = (h_out + padding_h - k_h) / stride_h;
                        int w_in = (w_out + padding_w - k_w) / stride_w;

                        if (d_in < 0 || d_in >= depth_in) continue;
                        if (h_in < 0 || h_in >= height_in) continue;
                        if (w_in < 0 || w_in >= width_in) continue;

                        int w_offset = i_c * out_channels * kernel_depth * kernel_height * kernel_width +
                                       o_c * kernel_depth * kernel_height * kernel_width +
                                       k_d * kernel_height * kernel_width +
                                       k_h * kernel_width +
                                       k_w;

                        int in_offset = b * in_channels * depth_in * height_in * width_in +
                                        i_c * depth_in * height_in * width_in +
                                        d_in * height_in * width_in +
                                        h_in * width_in +
                                        w_in;

                        sum += input[in_offset] * weight[w_offset];
                    }
                }
            }
        }

        int out_offset = b * out_channels * depth_out * height_out * width_out +
                         o_c * depth_out * height_out * width_out +
                         d_out * height_out * width_out +
                         h_out * width_out +
                         w_out;

        output[out_offset] = sum;
    }
}

at::Tensor conv_transpose3d_cuda(at::Tensor input, at::Tensor weight,
                                int kernel_depth, int kernel_height, int kernel_width,
                                int stride_d, int stride_h, int stride_w,
                                int padding_d, int padding_h, int padding_w,
                                int output_padding_d, int output_padding_h, int output_padding_w) {
    auto input_size = input.sizes();
    int batch_size = input_size[0];
    int in_channels = input_size[1];
    int depth_in = input_size[2];
    int height_in = input_size[3];
    int width_in = input_size[4];

    int out_channels = weight.size(1);
    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    auto output = at::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int threads = 256;
    const int elements = batch_size * out_channels * depth_out * height_out * width_out;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            depth_in,
            height_in,
            width_in,
            depth_out,
            height_out,
            width_out,
            kernel_depth,
            kernel_height,
            kernel_width,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            output_padding_d,
            output_padding_h,
            output_padding_w
        );
    }));

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_depth, int kernel_height, int kernel_width,
                                   int stride_d, int stride_h, int stride_w,
                                   int padding_d, int padding_h, int padding_w,
                                   int output_padding_d, int output_padding_h, int output_padding_w);
"""

conv_transpose3d_cuda = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x):
        conv = self.conv_transpose3d
        weight = conv.weight
        stride_d, stride_h, stride_w = conv.stride
        padding_d, padding_h, padding_w = conv.padding
        output_padding_d, output_padding_h, output_padding_w = conv.output_padding
        kernel_size_d, kernel_size_h, kernel_size_w = conv.kernel_size

        return conv_transpose3d_cuda.conv_transpose3d_cuda(
            x,
            weight,
            kernel_size_d, kernel_size_h, kernel_size_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w
        )