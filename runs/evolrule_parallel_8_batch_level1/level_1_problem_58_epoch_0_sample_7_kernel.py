import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel source code
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ inline int get_x_index(int n, int c_in, int d, int h, int w,
                                 int in_channels, int depth, int height, int width) {
    return n * in_channels * depth * height * width +
           c_in * depth * height * width +
           d * height * width +
           h * width +
           w;
}

__device__ inline int get_out_index(int n, int c_out, int d, int h, int w,
                                   int out_channels, int depth, int height, int width) {
    return n * out_channels * depth * height * width +
           c_out * depth * height * width +
           d * height * width +
           h * width +
           w;
}

__global__ void conv_transpose3d_kernel(
    const float* x,
    const float* weight,
    float* out,
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
    int output_padding_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out) {
        return;
    }

    int w_out = idx % width_out;
    int h_out = (idx / width_out) % height_out;
    int d_out = (idx / (width_out * height_out)) % depth_out;
    int c_out = (idx / (width_out * height_out * depth_out)) % out_channels;
    int n = idx / (width_out * height_out * depth_out * out_channels);

    float val = 0.0f;

    int in_per_group = in_channels / groups;
    int out_per_group = out_channels / groups;

    for (int g = 0; g < groups; ++g) {
        int start_out = g * out_per_group;
        int end_out = (g + 1) * out_per_group;
        if (c_out < start_out || c_out >= end_out) {
            continue;
        }

        int c_out_in_group = c_out - start_out;

        for (int c_in = g * in_per_group; c_in < (g + 1) * in_per_group; ++c_in) {
            for (int kd = 0; kd < kernel_depth; ++kd) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int d_in = (d_out - kd + 2 * padding_d - output_padding_d) / stride_d;
                        int h_in = (h_out - kh + 2 * padding_h - output_padding_h) / stride_h;
                        int w_in = (w_out - kw + 2 * padding_w - output_padding_w) / stride_w;

                        if (d_in < 0 || d_in >= depth_in || h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in) {
                            continue;
                        }

                        int in_channel_offset = c_in;
                        int out_channel_offset = c_out_in_group;
                        int weight_offset = in_channel_offset * (out_per_group * kernel_depth * kernel_height * kernel_width) +
                                            out_channel_offset * (kernel_depth * kernel_height * kernel_width) +
                                            kd * kernel_height * kernel_width +
                                            kh * kernel_width +
                                            kw;

                        float w_val = weight[weight_offset];
                        int x_idx = get_x_index(n, c_in, d_in, h_in, w_in, in_channels, depth_in, height_in, width_in);
                        float x_val = x[x_idx];

                        val += w_val * x_val;
                    }
                }
            }
        }
    }

    int out_idx = get_out_index(n, c_out, d_out, h_out, w_out, out_channels, depth_out, height_out, width_out);
    out[out_idx] = val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int groups
) {
    auto device = x.device();
    TORCH_CHECK(weight.device() == device, "weight must be on the same device as input");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int depth_in = x.size(2);
    int height_in = x.size(3);
    int width_in = x.size(4);

    int out_channels = weight.size(1) * groups;
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    int depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    int height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    auto out = torch::empty({batch_size, out_channels, depth_out, height_out, width_out}, x.options());

    const int threads_per_block = 256;
    int num_elements = batch_size * out_channels * depth_out * height_out * width_out;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
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
        output_padding_w,
        groups
    );

    cudaDeviceSynchronize();

    return out;
}
"""

cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int groups
);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        out = conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            output_padding_d,
            output_padding_h,
            output_padding_w,
            self.groups
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1, 1)
            out = out + bias_view

        return out