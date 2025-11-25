import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for transposed 3D convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out) {
        return;
    }

    int n = idx / (out_channels * depth_out * height_out * width_out);
    int rem = idx % (out_channels * depth_out * height_out * width_out);
    int o_c = rem / (depth_out * height_out * width_out);
    rem %= (depth_out * height_out * width_out);
    int d_out = rem / (height_out * width_out);
    rem %= (height_out * width_out);
    int h_out = rem / width_out;
    int w_out = rem % width_out;

    int out_channels_per_group = out_channels / groups;
    int group = o_c / out_channels_per_group;
    int o_c_in_group = o_c % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;
    int start_in_channel = group * in_channels_per_group;
    int end_in_channel = (group + 1) * in_channels_per_group;

    float acc = 0.0f;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int d_in = (d_out + padding_d - kd) / stride_d;
                int h_in = (h_out + padding_h - kh) / stride_h;
                int w_in = (w_out + padding_w - kw) / stride_w;

                if (d_in >= 0 && d_in < depth_in &&
                    h_in >= 0 && h_in < height_in &&
                    w_in >= 0 && w_in < width_in) {
                    for (int i_c = start_in_channel; i_c < end_in_channel; ++i_c) {
                        int weight_offset = i_c * out_channels_per_group * kernel_d * kernel_h * kernel_w +
                                           o_c_in_group * kernel_d * kernel_h * kernel_w +
                                           kd * kernel_h * kernel_w +
                                           kh * kernel_w +
                                           kw;
                        float w_val = weight[weight_offset];

                        int input_offset = n * in_channels * depth_in * height_in * width_in +
                                          i_c * depth_in * height_in * width_in +
                                          d_in * height_in * width_in +
                                          h_in * width_in +
                                          w_in;
                        float in_val = input[input_offset];

                        acc += in_val * w_val;
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[o_c];
    }

    output[idx] = acc;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
) {
    const int threads_per_block = 256;
    int num_elements = batch_size * out_channels * depth_out * height_out * width_out;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose3d_cpp_source = """
#include <vector>
#include <torch/extension.h>

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
);
"""

# Compile the CUDA kernel
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
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
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.output_padding = tuple(output_padding)
        self.groups = groups

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2]
        ))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters as per PyTorch's default initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.cuda_conv = conv_transpose3d

    def forward(self, x):
        batch_size, _, depth_in, height_in, width_in = x.size()
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        op_d, op_h, op_w = self.output_padding

        depth_out = (depth_in - 1) * stride_d + kernel_d - 2 * padding_d + op_d
        height_out = (height_in - 1) * stride_h + kernel_h - 2 * padding_h + op_h
        width_out = (width_in - 1) * stride_w + kernel_w - 2 * padding_w + op_w

        output = torch.empty(
            batch_size,
            self.out_channels,
            depth_out,
            height_out,
            width_out,
            device=x.device,
            dtype=x.dtype
        )

        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)

        self.cuda_conv(
            x, self.weight, bias,
            output,
            batch_size, self.in_channels, self.out_channels,
            depth_in, height_in, width_in,
            depth_out, height_out, width_out,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            op_d, op_h, op_w,
            self.groups
        )

        return output