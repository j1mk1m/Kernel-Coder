import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

static inline int compute_output_dim(int input_size, int kernel_size, int stride, int padding, int output_padding, int dilation) {
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
}

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width,
    int stride, int padding, int output_padding, int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) {
        return;
    }

    int b = idx / (out_channels * output_depth * output_height * output_width);
    int remaining = idx % (out_channels * output_depth * output_height * output_width);
    int c_out = remaining / (output_depth * output_height * output_width);
    remaining %= (output_depth * output_height * output_width);
    int d_out = remaining / (output_height * output_width);
    remaining %= (output_height * output_width);
    int h_out = remaining / output_width;
    int w_out = remaining % output_width;

    float sum = 0.0;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Compute input indices
                    int d_in = d_out - kd;
                    int h_in = h_out - kh;
                    int w_in = w_out - kw;

                    if (d_in < 0 || d_in >= input_depth ||
                        h_in < 0 || h_in >= input_height ||
                        w_in < 0 || w_in >= input_width) {
                        continue;
                    }

                    // Compute weight offset
                    int weight_offset = c_in * out_channels * kernel_size * kernel_size * kernel_size +
                                        c_out * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size +
                                        kh * kernel_size + kw;

                    float w = weight[weight_offset];

                    // Compute input offset
                    int input_offset = b * in_channels * input_depth * input_height * input_width +
                                      c_in * input_depth * input_height * input_width +
                                      d_in * input_height * input_width +
                                      h_in * input_width + w_in;

                    sum += input[input_offset] * w;
                }
            }
        }
    }

    // Write to output
    int output_offset = b * out_channels * output_depth * output_height * output_width +
                        c_out * output_depth * output_height * output_width +
                        d_out * output_height * output_width +
                        h_out * output_width + w_out;

    output[output_offset] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    const int out_channels = weight.size(1);
    const int output_depth = compute_output_dim(input_depth, kernel_size, stride, padding, output_padding, dilation);
    const int output_height = compute_output_dim(input_height, kernel_size, stride, padding, output_padding, dilation);
    const int output_width = compute_output_dim(input_width, kernel_size, stride, padding, output_padding, dilation);

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;

    dim3 threads_per_block(256);
    dim3 num_blocks((total_elements + threads_per_block.x - 1) / threads_per_block.x);

    conv_transpose3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, kernel_size,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        stride, padding, output_padding, dilation
    );

    return output;
}
"""

# Load the CUDA extension
conv_transpose3d_cuda = load_inline(
    name="conv_transpose3d_cuda",
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
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

        # Initialize weights with correct shape (in_channels, out_channels, kernel_size, ...)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        if self.bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using the same method as PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        output = conv_transpose3d_cuda.conv_transpose3d_cuda(
            x,
            self.weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation
        )
        if self.bias:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)
        return output