import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_h * output_w)
        return;

    int b = idx / (out_channels * output_h * output_w);
    int remaining = idx % (out_channels * output_h * output_w);
    int oc = remaining / (output_h * output_w);
    int remaining2 = remaining % (output_h * output_w);
    int oh = remaining2 / output_w;
    int ow = remaining2 % output_w;

    float sum = 0.0;

    int group_out_channels = out_channels / groups;
    int g = oc / group_out_channels;
    int group_in_channels = in_channels / groups;
    int start_ich = g * group_in_channels;

    for (int ich = start_ich; ich < start_ich + group_in_channels; ++ich) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride + kh * dilation - padding;
                int iw = ow * stride + kw * dilation - padding;
                if (ih < 0 || ih >= height || iw < 0 || iw >= width)
                    continue;

                int input_offset = b * in_channels * height * width
                    + ich * height * width
                    + ih * width + iw;
                float input_val = input[input_offset];

                int local_ich = ich - start_ich;
                int weight_offset = oc * group_in_channels * kernel_size * kernel_size
                    + local_ich * kernel_size * kernel_size
                    + kh * kernel_size + kw;
                float weight_val = weight[weight_offset];

                sum += input_val * weight_val;
            }
        }
    }

    if (bias != nullptr)
        sum += bias[oc];

    int output_offset = b * out_channels * output_h * output_w
        + oc * output_h * output_w
        + oh * output_w + ow;
    output[output_offset] = sum;
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    int num_elements = batch_size * out_channels * output_h * output_w;
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    custom_conv2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        stride, padding, dilation, groups,
        output_h, output_w
    );

    cudaDeviceSynchronize();
    return output;
}
"""

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources="",
    cuda_sources=custom_conv_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_tensor = self.bias if self.bias is not None else torch.tensor([])
        return custom_conv.custom_conv2d_cuda(
            x, self.weight, bias_tensor,
            self.stride, self.padding, self.dilation, self.groups
        )