import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int depth_out,
    int height_out,
    int width_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out) return;

    int w_out = idx % width_out;
    int rem = idx / width_out;
    int h_out = rem % height_out;
    rem = rem / height_out;
    int d_out = rem % depth_out;
    rem = rem / depth_out;
    int out_c = rem % out_channels;
    rem = rem / out_channels;
    int n = rem;

    float acc = 0.0;
    int in_c_per_group = in_channels / groups;

    for (int g = 0; g < groups; ++g) {
        for (int k_d = 0; k_d < kernel_d; ++k_d) {
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    int d_in = d_out * stride_d - padding_d + k_d * dilation_d;
                    int h_in = h_out * stride_h - padding_h + k_h * dilation_h;
                    int w_in = w_out * stride_w - padding_w + k_w * dilation_w;

                    if (d_in < 0 || d_in >= depth_in || h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in) {
                        continue;
                    }

                    for (int i_c = 0; i_c < in_c_per_group; ++i_c) {
                        int in_channel = g * in_c_per_group + i_c;

                        int input_offset = n * in_channels * depth_in * height_in * width_in +
                                           in_channel * depth_in * height_in * width_in +
                                           d_in * height_in * width_in +
                                           h_in * width_in +
                                           w_in;

                        int weight_offset = out_c * in_c_per_group * kernel_d * kernel_h * kernel_w +
                                           i_c * kernel_d * kernel_h * kernel_w +
                                           k_d * kernel_h * kernel_w +
                                           k_h * kernel_w +
                                           k_w;

                        acc += input[input_offset] * weight[weight_offset];
                    }
                }
            }
        }
    }

    if (bias) {
        acc += bias[out_c];
    }

    int output_offset = n * out_channels * depth_out * height_out * width_out +
                        out_c * depth_out * height_out * width_out +
                        d_out * height_out * width_out +
                        h_out * width_out +
                        w_out;

    output[output_offset] = acc;
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 int stride_d, int stride_h, int stride_w,
                                 int padding_d, int padding_h, int padding_w,
                                 int dilation_d, int dilation_h, int dilation_w,
                                 int groups) {
    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    int out_channels = weight.size(0);

    int depth_out = (D_in + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int height_out = (H_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int width_out = (W_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, out_channels, depth_out, height_out, width_out},
                              input.options());

    int num_threads = N * out_channels * depth_out * height_out * width_out;
    int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;

    // Launch kernel
    conv3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, out_channels, D_in, H_in, W_in,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        depth_out, height_out, width_out
    );

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                 int stride_d, int stride_h, int stride_w,
                                 int padding_d, int padding_h, int padding_w,
                                 int dilation_d, int dilation_h, int dilation_w,
                                 int groups);
"""

conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation

        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)

        output = conv3d_cuda.conv3d_forward_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias_tensor.contiguous(),
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w,
            self.groups
        )
        return output

def get_inputs():
    x = torch.rand(8, 3, 16, 128, 128).cuda()
    return [x]

def get_init_inputs():
    return [3, 64, (3,5,7)]  # in_channels, out_channels, kernel_size