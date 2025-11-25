import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel code
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int D_out,
    int H_out,
    int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * D_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (W_out * H_out * D_out * C_out);

    scalar_t sum = 0.0;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < kernel_size_d; ++kd) {
            for (int kh = 0; kh < kernel_size_h; ++kh) {
                for (int kw = 0; kw < kernel_size_w; ++kw) {
                    int d_in = (d_out + padding_d - kd*dilation_d - output_padding_d) / stride_d;
                    int h_in = (h_out + padding_h - kh*dilation_h - output_padding_h) / stride_h;
                    int w_in = (w_out + padding_w - kw*dilation_w - output_padding_w) / stride_w;

                    if (d_in < 0 || d_in >= D_in || h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) {
                        continue;
                    }

                    int kernel_offset = kd * kernel_size_h * kernel_size_w + kh * kernel_size_w + kw;
                    int weight_idx = c_in * C_out * kernel_size_d * kernel_size_h * kernel_size_w +
                                    c_out * kernel_size_d * kernel_size_h * kernel_size_w +
                                    kernel_offset;

                    scalar_t w = weight[weight_idx];

                    int input_offset = n * C_in * D_in * H_in * W_in +
                                      c_in * D_in * H_in * W_in +
                                      d_in * H_in * W_in +
                                      h_in * W_in +
                                      w_in;
                    scalar_t in_val = input[input_offset];

                    sum += w * in_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int output_offset = n * C_out * D_out * H_out * W_out +
                        c_out * D_out * H_out * W_out +
                        d_out * H_out * W_out +
                        h_out * W_out +
                        w_out;

    output[output_offset] = sum;
}

std::tuple<torch::Tensor> conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    auto C_out = weight.size(1);
    auto kernel_size_d = weight.size(2);
    auto kernel_size_h = weight.size(3);
    auto kernel_size_w = weight.size(4);

    int D_out = (D_in - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + output_padding_d + 1;
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + output_padding_h + 1;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + output_padding_w + 1;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    int total_threads = N * C_out * D_out * H_out * W_out;
    int threads_per_block = 1024;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<float><<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        D_out, H_out, W_out
    );

    cudaDeviceSynchronize();
    return {output};
}
"""

conv_transpose3d = load_inline(
    name='conv_transpose3d',
    cuda_sources=conv_transpose3d_source,
    functions=['conv_transpose3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias

        kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size_tuple))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding
        dilation_d, dilation_h, dilation_w = self.dilation

        if self.bias is not None:
            bias = self.bias
        else:
            bias = torch.empty(0)  # Pass an empty tensor if no bias

        output = conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            bias,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            dilation_d, dilation_h, dilation_w,
            self.groups
        )[0]

        return output