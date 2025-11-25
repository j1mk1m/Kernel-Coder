import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups,
    bool has_bias,
    const float* bias,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    int output_size = batch_size * out_channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (out_channels * output_height * output_width);

    float acc = 0.0;

    int C_out_per_group = out_channels / groups;
    int group = c_out / C_out_per_group;
    int c_out_in_group = c_out % C_out_per_group;

    int in_channels_per_group = in_channels / groups;
    int c_in_start = group * in_channels_per_group;

    for (int c_in = c_in_start; c_in < c_in_start + in_channels_per_group; c_in++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in = (h_out + padding_h - kh - output_padding_h) / stride_h;
                int w_in = (w_out + padding_w - kw - output_padding_w) / stride_w;

                if (h_in >= 0 && h_in < input_height &&
                    w_in >= 0 && w_in < input_width) {

                    int weight_offset = (c_in - c_in_start) * (C_out_per_group * kernel_h * kernel_w);
                    weight_offset += c_out_in_group * kernel_h * kernel_w;
                    weight_offset += kh * kernel_w + kw;

                    float w_val = weight[weight_offset];

                    int input_offset = n * in_channels * input_height * input_width;
                    input_offset += c_in * input_height * input_width;
                    input_offset += h_in * input_width + w_in;

                    float in_val = input[input_offset];

                    acc += in_val * w_val;
                }
            }
        }
    }

    if (has_bias) {
        acc += bias[c_out];
    }

    int output_offset = n * out_channels * output_height * output_width;
    output_offset += c_out * output_height * output_width;
    output_offset += h_out * output_width + w_out;

    output[output_offset] = acc;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   torch::Tensor bias,
                                   int stride_h,
                                   int stride_w,
                                   int padding_h,
                                   int padding_w,
                                   int output_padding_h,
                                   int output_padding_w,
                                   int kernel_h,
                                   int kernel_w,
                                   int groups,
                                   bool has_bias,
                                   int input_height,
                                   int input_width,
                                   int output_height,
                                   int output_width) {
    auto output = torch::empty({input.size(0),
                               weight.size(1)*groups,
                               output_height,
                               output_width},
                               input.options());

    const int threads_per_block = 256;
    int num_elements = input.size(0) * output_height * output_width * weight.size(1)*groups;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;

    conv_transpose2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        weight.size(0),
        weight.size(1)*groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        has_bias,
        bias_ptr,
        input_height,
        input_width,
        output_height,
        output_width
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w, int kernel_h, int kernel_w, int groups, bool has_bias, int input_height, int input_width, int output_height, int output_width);"
)

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias

        # Compute kernel dimensions
        self.kernel_h, self.kernel_w = kernel_size

        # Check divisibility by groups
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        # Initialize weight and bias
        weight_shape = (in_channels, out_channels // groups, self.kernel_h, self.kernel_w)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He initialization

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Load the CUDA kernel
        self.cuda_conv_transpose2d = conv_transpose2d

    def forward(self, x):
        batch_size, _, input_height, input_width = x.size()

        # Compute output dimensions
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_h + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_w + self.output_padding

        # Prepare parameters for the CUDA kernel
        has_bias = self.bias_flag
        bias = self.bias if has_bias else None

        # Call the CUDA kernel
        output = self.cuda_conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            bias,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.output_padding,
            self.output_padding,
            self.kernel_h,
            self.kernel_w,
            self.groups,
            has_bias,
            input_height,
            input_width,
            output_height,
            output_width
        )

        return output