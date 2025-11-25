import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void conv3d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size,
    int padding,
    int stride,
    int dilation,
    int groups,
    int output_depth,
    int output_height,
    int output_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    if (index >= total_elements) {
        return;
    }

    // Compute output indices
    int w_out = index % output_width;
    int h_out_offset = index / output_width;
    int h_out = h_out_offset % output_height;
    int d_out_offset = h_out_offset / output_height;
    int d_out = d_out_offset % output_depth;
    int co_offset = d_out_offset / output_depth;
    int c_out = co_offset % out_channels;
    int n = co_offset / out_channels;

    // Compute group
    int out_channels_per_group = out_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_in_group = c_out % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;
    int c_in_start = group * in_channels_per_group;

    float acc = 0.0f;

    // Iterate over input channels in the group
    for (int c_in_in_group = 0; c_in_in_group < in_channels_per_group; ++c_in_in_group) {
        int c_in = c_in_start + c_in_in_group;

        // Iterate over kernel spatial dimensions
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Compute input spatial indices
                    int d_in = d_out * stride - padding + kd * dilation;
                    int h_in = h_out * stride - padding + kh * dilation;
                    int w_in = w_out * stride - padding + kw * dilation;

                    // Check boundaries
                    if (d_in < 0 || d_in >= depth) continue;
                    if (h_in < 0 || h_in >= height) continue;
                    if (w_in < 0 || w_in >= width) continue;

                    // Compute input offset
                    int input_offset = n * in_channels * depth * height * width +
                                       c_in * depth * height * width +
                                       d_in * height * width +
                                       h_in * width +
                                       w_in;
                    float input_val = input[input_offset];

                    // Compute weight offset
                    int weight_c_out_base = group * out_channels_per_group + c_out_in_group;
                    int weight_offset = weight_c_out_base * in_channels_per_group * kernel_size * kernel_size * kernel_size +
                                        c_in_in_group * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size +
                                        kh * kernel_size +
                                        kw;

                    float weight_val = weight[weight_offset];

                    acc += input_val * weight_val;
                }
            }
        }
    }

    // Add bias
    if (bias) {
        acc += bias[c_out];
    }

    // Compute output offset
    int output_offset = n * out_channels * output_depth * output_height * output_width +
                        c_out * output_depth * output_height * output_width +
                        d_out * output_height * output_width +
                        h_out * output_width +
                        w_out;

    output[output_offset] = acc;
}

torch::Tensor conv3d_forward(
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
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_depth = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int total_threads = batch_size * out_channels * output_depth * output_height * output_width;
    int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    conv3d_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        padding,
        stride,
        dilation,
        groups,
        output_depth,
        output_height,
        output_width
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
"""

conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward"],
    verbose=True,
    extra_cflags=["-DDEBUG"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        # Initialize weight
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size
        ))
        # Initialize bias
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

        # The CUDA function is loaded inline
        self.conv3d_forward = conv3d.conv3d_forward

    def forward(self, x):
        x = x.contiguous()
        weight = self.weight.contiguous()
        if self.bias is not None:
            bias = self.bias.contiguous()
        else:
            bias = torch.empty(0, device=x.device)

        return self.conv3d_forward(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]