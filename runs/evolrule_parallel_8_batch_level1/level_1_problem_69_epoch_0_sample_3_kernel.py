import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_h * out_w)
        return;

    int n = idx / (out_channels * out_h * out_w);
    int remaining = idx % (out_channels * out_h * out_w);
    int c = remaining / (out_h * out_w);
    int yx = remaining % (out_h * out_w);
    int y = yx / out_w;
    int x = yx % out_w;

    scalar_t sum = 0.0;

    for (int g = 0; g < groups; ++g) {
        int in_channels_per_group = in_channels / groups;
        int out_channels_per_group = out_channels / groups;
        int in_c_start = g * in_channels_per_group;
        int out_c_start = g * out_channels_per_group;
        if (c < out_c_start || c >= out_c_start + out_channels_per_group)
            continue;
        int local_out_c = c - out_c_start;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_y = (y - kh * dilation_h + padding_h - output_padding_h) / stride_h;
                int input_x = (x - kw * dilation_w + padding_w - output_padding_w) / stride_w;
                if (input_y < 0 || input_y >= in_h || input_x < 0 || input_x >= in_w)
                    continue;

                for (int in_c = in_c_start; in_c < in_c_start + in_channels_per_group; ++in_c) {
                    // Compute weight index within the group
                    int w_offset = in_c * out_channels_per_group * kernel_h * kernel_w +
                                   local_out_c * kernel_h * kernel_w +
                                   kh * kernel_w + kw;
                    scalar_t w = weight[w_offset];

                    // Compute input value
                    int in_offset = n * in_channels * in_h * in_w +
                                    in_c * in_h * in_w +
                                    input_y * in_w + input_x;
                    scalar_t in_val = input[in_offset];

                    sum += w * in_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c];
    }

    int out_offset = n * out_channels * out_h * out_w +
                     c * out_h * out_w +
                     y * out_w + x;
    output[out_offset] = sum;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   torch::Tensor bias,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int output_padding_h, int output_padding_w,
                                   int dilation_h, int dilation_w,
                                   int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Compute output dimensions
    int out_h = (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    int out_w = (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    int threads_per_block = 256;
    int num_blocks = (batch_size * out_channels * out_h * out_w + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_padding_h,
            output_padding_w,
            dilation_h,
            dilation_w,
            groups
        );
    }));

    return output;
}
"""

conv_transpose2d_cpp = """
#include <torch/extension.h>

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   torch::Tensor bias,
                                   int stride_h, int stride_w,
                                   int padding_h, int padding_w,
                                   int output_padding_h, int output_padding_w,
                                   int dilation_h, int dilation_w,
                                   int groups);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias

        # Initialize weights and bias
        kh, kw = kernel_size
        weight_shape = (in_channels, out_channels // groups, kh, kw)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters (as per PyTorch's default initialization)
        self.reset_parameters()

        # The CUDA function is already compiled
        self.conv_transpose2d_cuda = conv_transpose2d

    def reset_parameters(self):
        # Use the same initialization as PyTorch's ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack parameters
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups
        bias = self.bias if self.bias is not None else torch.empty(0)

        # Call the CUDA kernel
        return self.conv_transpose2d_cuda(
            x, self.weight, bias,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            dilation_h, dilation_w,
            groups
        )