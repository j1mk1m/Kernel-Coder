import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_forward_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int dilation_depth, int dilation_height, int dilation_width,
    int groups,
    bool has_bias,
    int output_depth, int output_height, int output_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_depth * output_height * output_width) {
        return;
    }

    int w_out = index % output_width;
    index /= output_width;
    int h_out = index % output_height;
    index /= output_height;
    int d_out = index % output_depth;
    index /= output_depth;
    int c_out = index % out_channels;
    int n = index / out_channels;

    float sum = 0.0f;
    for (int kd = 0; kd < kernel_depth; ++kd) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int d_in = d_out * stride_depth - padding_depth + kd * dilation_depth;
                int h_in = h_out * stride_height - padding_height + kh * dilation_height;
                int w_in = w_out * stride_width - padding_width + kw * dilation_width;

                if (d_in < 0 || d_in >= input_depth || 
                    h_in < 0 || h_in >= input_height || 
                    w_in < 0 || w_in >= input_width) {
                    continue;
                }

                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int input_offset = (
                        n * in_channels + c_in) * input_depth * input_height * input_width 
                        + d_in * input_height * input_width 
                        + h_in * input_width 
                        + w_in;

                    int group_out = c_out / (out_channels / groups);
                    int per_group_out = c_out % (out_channels / groups);
                    int per_group_in = (c_in % (in_channels / groups));

                    int weight_offset = (
                        group_out * (out_channels / groups * in_channels / groups * kernel_depth * kernel_height * kernel_width) +
                        per_group_out * (in_channels / groups * kernel_depth * kernel_height * kernel_width) +
                        per_group_in * (kernel_depth * kernel_height * kernel_width) +
                        kd * kernel_height * kernel_width +
                        kh * kernel_width +
                        kw);

                    sum += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[c_out];
    }

    int output_offset = (
        n * out_channels + c_out) * output_depth * output_height * output_width 
        + d_out * output_height * output_width 
        + h_out * output_width 
        + w_out;

    output[output_offset] = sum;
}

torch::Tensor custom_conv3d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    bool has_bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0) * groups;
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    int output_depth = (input_depth + 2 * padding[0] - dilation[0] * (kernel_depth - 1) - 1) / stride[0] + 1;
    int output_height = (input_height + 2 * padding[1] - dilation[1] * (kernel_height - 1) - 1) / stride[1] + 1;
    int output_width = (input_width + 2 * padding[2] - dilation[2] * (kernel_width - 1) - 1) / stride[2] + 1;

    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    conv3d_forward_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        (has_bias) ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth, input_height, input_width,
        kernel_depth, kernel_height, kernel_width,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups,
        has_bias,
        output_depth, output_height, output_width
    );

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor custom_conv3d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    bool has_bias
);
"""

custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["custom_conv3d"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False):
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        self.weight = nn.Parameter(torch.empty(
            out_channels // groups,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2]
        ))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return custom_conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.has_bias
        )