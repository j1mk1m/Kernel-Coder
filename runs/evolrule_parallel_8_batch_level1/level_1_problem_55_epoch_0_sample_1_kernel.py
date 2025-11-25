import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom Conv2D kernel code
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width,
    int groups
) {
    int batch = threadIdx.x;
    int out_h = blockIdx.x;
    int out_w = blockIdx.y;
    if (batch >= batch_size || out_h >= output_height || out_w >= output_width) {
        return;
    }

    for (int out_channel = 0; out_channel < out_channels; ++out_channel) {
        float sum = 0.0f;
        int out_channels_per_group = out_channels / groups;
        int group = out_channel / out_channels_per_group;
        int in_channels_per_group = in_channels / groups;
        int in_ch_start = group * in_channels_per_group;
        int in_ch_end = (group + 1) * in_channels_per_group;

        for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_h = out_h * stride + kh * dilation - padding;
                    int input_w = out_w * stride + kw * dilation - padding;
                    if (input_h >= 0 && input_h < input_height && input_w >= 0 && input_w < input_width) {
                        int input_offset = batch * in_channels * input_height * input_width +
                            in_ch * input_height * input_width +
                            input_h * input_width + input_w;
                        float input_val = input[input_offset];

                        int local_in_channel = in_ch - in_ch_start;
                        int weight_offset = out_channel * in_channels_per_group * kernel_size * kernel_size +
                            local_in_channel * kernel_size * kernel_size +
                            kh * kernel_size + kw;
                        float weight_val = weight[weight_offset];

                        sum += input_val * weight_val;
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[out_channel];
        }

        int output_offset = batch * out_channels * output_height * output_width +
            out_channel * output_height * output_width +
            out_h * output_width + out_w;
        output[output_offset] = sum;
    }
}

torch::Tensor custom_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    int output_height,
    int output_width,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int input_height = input.size(2);
    int input_width = input.size(3);

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 block(batch_size);
    dim3 grid(output_height, output_width);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv2d_forward_cuda", ([&] {
        custom_conv2d_forward<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            output_height,
            output_width,
            groups
        );
    }));

    return output;
}
"""

conv2d_header = """
torch::Tensor custom_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    int output_height,
    int output_width,
    int groups
);
"""

# Load the CUDA extension
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_header,
    cuda_sources=conv2d_source,
    functions=["custom_conv2d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        kernel_size = self.kernel_size
        groups = self.groups
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else torch.empty(0)
        # Compute output dimensions
        batch_size = x.size(0)
        in_channels = x.size(1)
        input_height = x.size(2)
        input_width = x.size(3)
        output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Call the custom CUDA kernel
        return custom_conv2d.custom_conv2d_forward_cuda(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            kernel_size,
            output_height,
            output_width,
            groups
        )