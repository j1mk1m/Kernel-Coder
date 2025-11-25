import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel source code
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) return;

    int batch = idx / (out_channels * output_depth * output_height * output_width);
    int oc = (idx % (out_channels * output_depth * output_height * output_width)) / (output_depth * output_height * output_width);
    int dz = (idx % (output_depth * output_height * output_width)) / (output_height * output_width);
    int dy = (idx % (output_height * output_width)) / output_width;
    int dx = idx % output_width;

    float acc = 0.0;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int id = (dz - kd * dilation + padding) / stride;
                    int ih = (dy - kh * dilation + padding) / stride;
                    int iw = (dx - kw * dilation + padding) / stride;

                    if (id >= 0 && id < input_depth &&
                        ih >= 0 && ih < input_height &&
                        iw >= 0 && iw < input_width) {

                        int weight_offset = ic * out_channels * kernel_size * kernel_size * kernel_size
                            + oc * kernel_size * kernel_size * kernel_size
                            + kd * kernel_size * kernel_size
                            + kh * kernel_size
                            + kw;
                        float w = weight[weight_offset];

                        int input_offset = batch * in_channels * input_depth * input_height * input_width
                            + ic * input_depth * input_height * input_width
                            + id * input_height * input_width
                            + ih * input_width
                            + iw;
                        float in_val = input[input_offset];

                        acc += in_val * w;
                    }
                }
            }
        }
    }

    int output_offset = batch * out_channels * output_depth * output_height * output_width
        + oc * output_depth * output_height * output_width
        + dz * output_height * output_width
        + dy * output_width
        + dx;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    int output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    int total_threads = batch_size * out_channels * output_depth * output_height * output_width;
    const int block_size = 256;
    dim3 blocks((total_threads + block_size - 1) / block_size);
    dim3 threads(block_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width
        );
    }));

    return output;
}
"""

# Define the header for the C++ code
conv_transpose3d_h = """
#include <torch/extension.h>
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);
"""

# Compile the CUDA code
conv_transpose3d_cuda = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources=conv_transpose3d_h,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_cuda.conv_transpose3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)