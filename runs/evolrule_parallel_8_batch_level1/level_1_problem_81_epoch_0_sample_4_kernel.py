import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv2d_forward(
    const scalar_t* input,
    const scalar_t* kernel,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height_out * width_out) return;

    int x_out = idx % width_out;
    int y_out = (idx / width_out) % height_out;
    int out_ch = (idx / (width_out * height_out)) % out_channels;
    int b = idx / (width_out * height_out * out_channels);

    scalar_t sum = 0.0;

    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int y_in = (y_out + padding - ky * dilation) / stride;
                int x_in = (x_out + padding - kx * dilation) / stride;

                if (y_in < 0 || y_in >= height_in || x_in < 0 || x_in >= width_in) continue;

                int kernel_idx = out_ch * in_channels * kernel_size * kernel_size +
                                 in_ch * kernel_size * kernel_size +
                                 ky * kernel_size + kx;
                scalar_t weight = kernel[kernel_idx];

                int input_offset = b * in_channels * height_in * width_in +
                                   in_ch * height_in * width_in +
                                   y_in * width_in + x_in;
                scalar_t val = input[input_offset];

                sum += val * weight;
            }
        }
    }

    int out_offset = b * out_channels * height_out * width_out +
                     out_ch * height_out * width_out +
                     y_out * width_out + x_out;
    output[out_offset] = sum;
}

at::Tensor transposed_conv2d_cuda(
    const at::Tensor& input,
    const at::Tensor& kernel,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int out_channels = kernel.size(0);
    int kernel_size = kernel.size(2);

    int height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    at::Tensor output = at::empty({batch_size, out_channels, height_out, width_out}, input.options());

    const int threads_per_block = 256;
    const int elements = batch_size * out_channels * height_out * width_out;
    const int blocks = (elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "transposed_conv2d_forward", ([&] {
        transposed_conv2d_forward<scalar_t><<<blocks, threads_per_block>>>(
            input.data<scalar_t>(),
            kernel.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_size, height_in, width_in,
            height_out, width_out,
            stride, padding, dilation
        );
    }));

    return output;
}
"""

transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources="""extern "C" {
        at::Tensor transposed_conv2d_cuda(const at::Tensor& input, const at::Tensor& kernel, int stride, int padding, int dilation);
    }""",
    cuda_sources=transposed_conv_source,
    functions=["transposed_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = transposed_conv.transposed_conv2d_cuda(
            x,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output