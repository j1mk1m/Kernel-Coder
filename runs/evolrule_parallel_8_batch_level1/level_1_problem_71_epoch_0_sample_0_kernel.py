import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int H_out,
    int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * H_out * W_out) {
        return;
    }

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (H_out * W_out)) % out_channels;
    int n = idx / (out_channels * H_out * W_out);

    scalar_t val = 0.0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = (h_out + 2 * padding - kh) / stride;
            int w_in = (w_out + 2 * padding - kw) / stride;

            if (h_in < 0 || h_in >= input_h || w_in < 0 || w_in >= input_w) {
                continue;
            }

            int group_id = c_out % groups;
            int in_group_size = in_channels / groups;
            int start_c_in = group_id * in_group_size;
            int end_c_in = (group_id + 1) * in_group_size;

            for (int c_in = start_c_in; c_in < end_c_in; ++c_in) {
                int out_group = c_out / groups;
                int weight_offset = c_in * (out_channels / groups) * kernel_size * kernel_size
                    + out_group * kernel_size * kernel_size
                    + kh * kernel_size + kw;

                int input_offset = n * in_channels * input_h * input_w
                    + c_in * input_h * input_w
                    + h_in * input_w + w_in;

                val += input[input_offset] * weight[weight_offset];
            }
        }
    }

    if (bias != nullptr) {
        val += bias[c_out];
    }

    int output_offset = n * out_channels * H_out * W_out
        + c_out * H_out * W_out
        + h_out * W_out + w_out;

    output[output_offset] = val;
}

at::Tensor custom_conv_transpose2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;

    int H_out = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;

    at::Tensor output = at::empty({batch_size, out_channels, H_out, W_out}, input.options());

    const int total_threads = batch_size * out_channels * H_out * W_out;
    const int block_size = 256;
    dim3 blocks((total_threads + block_size - 1) / block_size, 1, 1);
    dim3 threads(block_size, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv_transpose2d_cuda", ([&] {
        custom_conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_h,
            input_w,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            H_out,
            W_out
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

custom_conv_cpp_source = """
at::Tensor custom_conv_transpose2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
);
"""

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=[custom_conv_cpp_source],
    cuda_sources=[custom_conv_source],
    functions=["custom_conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[],
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
        self.bias_flag = bias

        weight_shape = (in_channels, out_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape).cuda())
        if self.bias_flag:
            self.bias = nn.Parameter(torch.empty(out_channels).cuda())
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.forward_function = custom_conv.custom_conv_transpose2d_cuda

    def forward(self, x):
        bias = self.bias if self.bias is not None else None
        return self.forward_function(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )