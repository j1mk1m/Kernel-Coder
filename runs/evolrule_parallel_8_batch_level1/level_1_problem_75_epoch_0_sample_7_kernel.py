import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) return;

    int b = idx / (out_channels * output_height * output_width);
    int rest = idx % (out_channels * output_height * output_width);
    int oc = rest / (output_height * output_width);
    rest %= (output_height * output_width);
    int h_out = rest / output_width;
    int w_out = rest % output_width;

    int out_channels_per_group = out_channels / groups;
    int g = oc / out_channels_per_group;
    int oc_group = oc % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;
    int in_start = g * in_channels_per_group;

    scalar_t sum = 0.0;

    for (int ic = 0; ic < in_channels_per_group; ++ic) {
        int in_channel = in_start + ic;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = (h_out - kh * dilation_h - padding_h) / stride_h;
                int w_in = (w_out - kw * dilation_w - padding_w) / stride_w;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int weight_offset = (in_channel) * out_channels_per_group * kernel_h * kernel_w
                        + oc_group * kernel_h * kernel_w
                        + kh * kernel_w + kw;
                    scalar_t w = weight[weight_offset];

                    int input_offset = b * in_channels * input_height * input_width
                        + in_channel * input_height * input_width
                        + h_in * input_width + w_in;
                    scalar_t in_val = input[input_offset];

                    sum += in_val * w;
                }
            }
        }
    }

    int out_offset = b * out_channels * output_height * output_width
        + oc * output_height * output_width
        + h_out * output_width + w_out;
    output[out_offset] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1) * groups;
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int total_elements = batch_size * out_channels * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = None
        kernel_h, kernel_w = kernel_size
        weight_shape = (in_channels, out_channels // groups, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(weight_shape, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups
        weight = self.weight

        output = conv_transpose2d.conv_transpose2d_cuda(
            x, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output