import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv_transpose2d(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w,
    int out_h, int out_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_h * out_w) return;

    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int oc = (idx / (out_h * out_w)) % out_channels;
    int b = idx / (out_channels * out_h * out_w);

    float acc = 0.0;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int numerator_h = h_out + pad_h - kh + out_pad_h;
                int in_h = numerator_h / stride_h;
                if (numerator_h % stride_h != 0) continue;

                int numerator_w = w_out + pad_w - kw + out_pad_w;
                int in_w = numerator_w / stride_w;
                if (numerator_w % stride_w != 0) continue;

                if (in_h < 0 || in_h >= input_h || in_w < 0 || in_w >= input_w)
                    continue;

                int in_idx = b * in_channels * input_h * input_w +
                            ic * input_h * input_w +
                            in_h * input_w + in_w;

                int wt_idx = oc * in_channels * kernel_h * kernel_w +
                            ic * kernel_h * kernel_w +
                            kh * kernel_w + kw;

                acc += weight[wt_idx] * input[in_idx];
            }
        }
    }

    int out_idx = b * out_channels * out_h * out_w +
                  oc * out_h * out_w +
                  h_out * out_w + w_out;

    output[out_idx] = acc;
}

torch::Tensor custom_conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = weight.size(0);

    int out_h = (input_h - 1)*stride_h - 2*pad_h + kernel_h + out_pad_h;
    int out_w = (input_w - 1)*stride_w - 2*pad_w + kernel_w + out_pad_w;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    int total_elements = batch_size * out_channels * out_h * out_w;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    custom_conv_transpose2d<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_h, input_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        out_h, out_w
    );

    return output;
}
"""

cpp_source = """
torch::Tensor custom_conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int out_pad_h, int out_pad_w);
"""

conv_transpose_cuda = load_inline(
    name="custom_conv_transpose",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["custom_conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.groups = groups
        self.bias = bias

        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups,
            kernel_size[0], kernel_size[1]
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # The custom CUDA function
        self.custom_conv_transpose = conv_transpose_cuda

    def forward(self, x):
        # Unpack parameters
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        out_pad_h, out_pad_w = self.output_padding
        kernel_h, kernel_w = self.kernel_size

        # Call the custom CUDA function
        output = self.custom_conv_transpose(
            x, self.weight,
            stride_h, stride_w,
            pad_h, pad_w,
            out_pad_h, out_pad_w
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output