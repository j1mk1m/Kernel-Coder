import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int out_channels, int in_channels, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int output_padding_h, int output_padding_w, int dilation_h, int dilation_w,
    int groups
) {
    const int batch_size = input.size(0);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int output_h = output.size(2);
    const int output_w = output.size(3);

    const int n = blockIdx.x;
    const int c_out = blockIdx.y;
    const int y_out = threadIdx.y + blockDim.y * blockIdx.z;
    const int x_out = threadIdx.x + blockDim.x * blockIdx.w;

    if (y_out >= output_h || x_out >= output_w) return;

    scalar_t val = 0;

    for (int g = 0; g < groups; ++g) {
        const int c_in_group = c_out / (out_channels / groups);
        const int in_channel_start = g * (in_channels / groups);
        const int out_channel_start = g * (out_channels / groups);

        for (int kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
            for (int kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                const int y_in = (y_out - kernel_y * dilation_h - padding_h + output_padding_h) / stride_h;
                const int x_in = (x_out - kernel_x * dilation_w - padding_w + output_padding_w) / stride_w;

                if (y_in < 0 || y_in >= input_h || x_in < 0 || x_in >= input_w) continue;

                const int w_y = kernel_h - 1 - kernel_y;
                const int w_x = kernel_w - 1 - kernel_x;
                const int weight_idx = out_channel_start + c_out % (out_channels / groups);

                for (int c_in = in_channel_start; c_in < in_channels / groups + in_channel_start; ++c_in) {
                    val += input[n][c_in][y_in][x_in] * weight[weight_idx][c_in - in_channel_start][w_y][w_x];
                }
            }
        }
    }

    output[n][c_out][y_out][x_out] = val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    const int output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    const dim3 threads(16, 16);
    dim3 blocks(
        1,
        out_channels,
        (output_h + threads.y - 1) / threads.y,
        (output_w + threads.x - 1) / threads.x
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            out_channels, in_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w,
            output_padding_h, output_padding_w, dilation_h, dilation_w,
            groups
        );
    }));

    return output;
}
"""

conv_transpose2d_cpp_source = """
#include <torch/extension.h>
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups
);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), output_padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )