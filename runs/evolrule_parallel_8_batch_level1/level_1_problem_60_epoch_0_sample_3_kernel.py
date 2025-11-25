import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code for custom 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define KERNEL_DEPTH 3
#define KERNEL_HEIGHT 5
#define KERNEL_WIDTH 7

__global__ void conv3d_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int stride,
    int padding,
    int dilation,
    int out_depth,
    int out_height,
    int out_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * out_depth * out_height * out_width) return;

    int batch = index / (out_channels * out_depth * out_height * out_width);
    int remaining = index % (out_channels * out_depth * out_height * out_width);
    int out_channel = remaining / (out_depth * out_height * out_width);
    remaining = remaining % (out_depth * out_height * out_width);
    int out_z = remaining / (out_height * out_width);
    remaining = remaining % (out_height * out_width);
    int out_y = remaining / out_width;
    int out_x = remaining % out_width;

    float sum = 0.0f;

    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int kernel_d = 0; kernel_d < KERNEL_DEPTH; ++kernel_d) {
            for (int kernel_h = 0; kernel_h < KERNEL_HEIGHT; ++kernel_h) {
                for (int kernel_w = 0; kernel_w < KERNEL_WIDTH; ++kernel_w) {
                    int input_z = out_z * stride + kernel_d * dilation - padding;
                    int input_y = out_y * stride + kernel_h * dilation - padding;
                    int input_x = out_x * stride + kernel_w * dilation - padding;

                    if (input_z < 0 || input_z >= in_depth) continue;
                    if (input_y < 0 || input_y >= in_height) continue;
                    if (input_x < 0 || input_x >= in_width) continue;

                    int input_offset = in_channel * in_depth * in_height * in_width
                        + input_z * in_height * in_width
                        + input_y * in_width
                        + input_x;

                    int weight_offset = out_channel * in_channels * KERNEL_DEPTH * KERNEL_HEIGHT * KERNEL_WIDTH
                        + in_channel * KERNEL_DEPTH * KERNEL_HEIGHT * KERNEL_WIDTH
                        + kernel_d * KERNEL_HEIGHT * KERNEL_WIDTH
                        + kernel_h * KERNEL_WIDTH
                        + kernel_w;

                    float input_val = input[batch * in_channels * in_depth * in_height * in_width + input_offset];
                    float weight_val = weights[weight_offset];
                    sum += input_val * weight_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[out_channel];
    }

    int output_offset = out_channel * out_depth * out_height * out_width
        + out_z * out_height * out_width
        + out_y * out_width
        + out_x;

    output[batch * out_channels * out_depth * out_height * out_width + output_offset] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weights, torch::optional<torch::Tensor> bias,
                         int stride, int padding, int dilation) {
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);

    auto out_channels = weights.size(0);

    int out_depth = (in_depth + 2 * padding - dilation * (KERNEL_DEPTH - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (KERNEL_HEIGHT - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (KERNEL_WIDTH - 1) - 1) / stride + 1;

    auto output = torch::empty({input.size(0), out_channels, out_depth, out_height, out_width},
                              input.options());

    int num_threads = output.numel();
    const int block_size = 256;
    const int grid_size = (num_threads + block_size - 1) / block_size;

    conv3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        input.size(0), in_channels, in_depth, in_height, in_width,
        out_channels, stride, padding, dilation,
        out_depth, out_height, out_width
    );

    return output;
}
"""

conv3d_cpp_source = """
#include <torch/extension.h>

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weights, torch::optional<torch::Tensor> bias,
                         int stride, int padding, int dilation);
"""

# Compile the CUDA code
conv3d = load_inline(
    name="conv3d",
    cuda_sources=conv3d_source,
    cpp_sources=conv3d_cpp_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        assert groups == 1, "Groups not supported in this custom kernel yet"

        kernel_depth, kernel_height, kernel_width = kernel_size
        assert kernel_depth == 3 and kernel_height == 5 and kernel_width == 7, "Custom kernel requires kernel_size=(3,5,7)"
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_depth, kernel_height, kernel_width))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv3d.conv3d_cuda(x, self.weight, self.bias, self.stride, self.padding, self.dilation)