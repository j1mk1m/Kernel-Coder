import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width)
        return;

    int b = idx / (out_channels * output_depth * output_height * output_width);
    int rem = idx % (out_channels * output_depth * output_height * output_width);
    int oc = rem / (output_depth * output_height * output_width);
    rem %= (output_depth * output_height * output_width);
    int od = rem / (output_height * output_width);
    rem %= (output_height * output_width);
    int oh = rem / output_width;
    int ow = rem % output_width;

    int group = oc / (out_channels / groups);
    int oc_in_group = oc % (out_channels / groups);

    float sum = 0.0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int id = (od - kd + padding) / stride;
                int ih = (oh - kh + padding) / stride;
                int iw = (ow - kw + padding) / stride;

                if (id < 0 || id >= input_depth ||
                    ih < 0 || ih >= input_height ||
                    iw < 0 || iw >= input_width) {
                    continue;
                }

                for (int ic_in_group = 0; ic_in_group < (in_channels / groups); ++ic_in_group) {
                    int ic = group * (in_channels / groups) + ic_in_group;

                    int w_idx = ic * (out_channels / groups) * kernel_size * kernel_size * kernel_size +
                                oc_in_group * kernel_size * kernel_size * kernel_size +
                                kd * kernel_size * kernel_size +
                                kh * kernel_size +
                                kw;

                    float w = weight[w_idx];

                    int in_offset = b * in_channels * input_depth * input_height * input_width +
                                    ic * input_depth * input_height * input_width +
                                    id * input_height * input_width +
                                    ih * input_width +
                                    iw;

                    float in_val = input[in_offset];

                    sum += w * in_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_offset = b * out_channels * output_depth * output_height * output_width +
                     oc * output_depth * output_height * output_width +
                     od * output_height * output_width +
                     oh * output_width +
                     ow;
    output[out_offset] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(1) * groups;
    const int kernel_size = weight.size(2);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int num_threads = 1024;
    const int num_blocks = (output.numel() + num_threads - 1) / num_threads;

    conv_transpose3d_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias.defined()) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        stride,
        padding,
        output_padding,
        groups);

    return output;
}
"""

transposed_conv3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding, int groups);"
)

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=transposed_conv3d_cpp_source,
    cuda_sources=transposed_conv3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias if self.conv_transpose3d.bias is not None else torch.empty(0)
        return conv_transpose3d_cuda(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )