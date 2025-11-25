import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int input_h, int input_w,
    int output_h, int output_w) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int n = tid / (out_channels * output_h * output_w);
    int c_out = (tid % (out_channels * output_h * output_w)) / (output_h * output_w);
    int y = (tid % (output_h * output_w)) / output_w;
    int x = tid % output_w;

    if (n >= batch_size || c_out >= out_channels || y >= output_h || x >= output_w) return;

    int group = c_out / (out_channels / groups);
    int c_out_in_group = c_out % (out_channels / groups);

    float acc = 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int y_input = (y - kh * dilation_h + padding_h) / stride_h;
            int x_input = (x - kw * dilation_w + padding_w) / stride_w;

            if (y_input < 0 || y_input >= input_h || x_input < 0 || x_input >= input_w)
                continue;

            for (int c_in_in_group = 0; c_in_in_group < in_channels / groups; ++c_in_in_group) {
                int c_in = group * (in_channels / groups) + c_in_in_group;

                int input_offset = n * in_channels * input_h * input_w +
                    c_in * input_h * input_w + y_input * input_w + x_input;
                float in_val = input[input_offset];

                int weight_offset = group * (out_channels / groups) * (in_channels / groups) * kernel_h * kernel_w +
                    c_out_in_group * (in_channels / groups) * kernel_h * kernel_w +
                    c_in_in_group * kernel_h * kernel_w + kh * kernel_w + kw;
                float w_val = weight[weight_offset];

                acc += in_val * w_val;
            }
        }
    }

    if (bias) acc += bias[c_out];

    int output_offset = n * out_channels * output_h * output_w +
        c_out * output_h * output_w + y * output_w + x;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int out_channels = weight.size(0) * groups;
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int output_h = (input_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int num_threads = batch_size * out_channels * output_h * output_w;
    int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        input_h, input_w,
        output_h, output_w
    );

    return output;
}
"""

conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d_cuda",
    cuda_sources=conv_transpose2d_cuda,
    functions=["conv_transpose2d_cuda_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(
            out_channels // groups,
            in_channels // groups,
            *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose2d_cuda.conv_transpose2d_cuda_forward(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.Tensor(),
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )