import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ kernel,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int C_in, int D, int H, int W,
    int C_out, int kernel_kH, int kernel_kW,
    int stride, int padding, int dilation,
    int groups,
    int D_out, int H_out, int W_out
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N * C_out * D_out * H_out * W_out) return;

    int w_out = tid % W_out;
    int h_out = (tid / W_out) % H_out;
    int d_out = (tid / (W_out * H_out)) % D_out;
    int c_out = (tid / (W_out * H_out * D_out)) % C_out;
    int n = tid / (C_out * D_out * H_out * W_out);

    int d_in = d_out * stride - padding;
    if (d_in < 0 || d_in >= D) return;

    int group = c_out / (C_out / groups);
    int start_c_in = group * (C_in / groups);
    int end_c_in = start_c_in + (C_in / groups);

    scalar_t acc = 0.0;
    for (int c_in = start_c_in; c_in < end_c_in; ++c_in) {
        for (int kh = 0; kh < kernel_kH; ++kh) {
            for (int kw = 0; kw < kernel_kW; ++kw) {
                int h_in = h_out * stride - padding + kh * dilation;
                int w_in = w_out * stride - padding + kw * dilation;

                if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;

                int input_offset = n * C_in * D * H * W +
                                   c_in * D * H * W +
                                   d_in * H * W +
                                   h_in * W +
                                   w_in;
                scalar_t input_val = input[input_offset];

                int c_in_group = c_in - start_c_in;
                int kernel_offset = c_out * (C_in / groups) * kernel_kH * kernel_kW +
                                    c_in_group * kernel_kH * kernel_kW +
                                    kh * kernel_kW +
                                    kw;
                scalar_t kernel_val = kernel[kernel_offset];

                acc += input_val * kernel_val;
            }
        }
    }

    if (bias) acc += bias[c_out];

    int output_offset = n * C_out * D_out * H_out * W_out +
                       c_out * D_out * H_out * W_out +
                       d_out * H_out * W_out +
                       h_out * W_out +
                       w_out;
    output[output_offset] = acc;
}

std::tuple<torch::Tensor, int, int, int> custom_conv3d_forward(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int C_out = kernel.size(0);
    int kernel_C_in_per_group = kernel.size(1);
    int kernel_kD = kernel.size(2);
    int kernel_kH = kernel.size(3);
    int kernel_kW = kernel.size(4);

    int D_out = (D + 2 * padding - (kernel_kD - 1)*dilation) / stride;
    int H_out = (H + 2 * padding - (kernel_kH - 1)*dilation) / stride;
    int W_out = (W + 2 * padding - (kernel_kW - 1)*dilation) / stride;

    torch::Tensor output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    int total_threads = N * C_out * D_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv3d_forward", ([&] {
        custom_conv3d_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            kernel.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, C_in, D, H, W,
            C_out, kernel_kH, kernel_kW,
            stride, padding, dilation,
            groups,
            D_out, H_out, W_out
        );
    }));

    return std::make_tuple(output, D_out, H_out, W_out);
}
"""

cpp_source = """
#include <torch/extension.h>
std::tuple<torch::Tensor, int, int, int> custom_conv3d_forward(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
"""

custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=cpp_source,
    cuda_sources=custom_conv3d_source,
    functions=["custom_conv3d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, 1, kernel_size, kernel_size
        ))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_param", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        bias = self.bias_param if self.bias_param is not None else torch.empty(0)
        output, _, _, _ = custom_conv3d.custom_conv3d_forward(
            x, self.weight, bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        return output

def get_inputs():
    batch_size = 16
    in_channels = 3
    height = 256
    width = 256
    depth = 10
    x = torch.randn(batch_size, in_channels, height, width, depth).cuda()
    return [x]

def get_init_inputs():
    return [3, 64, 3]