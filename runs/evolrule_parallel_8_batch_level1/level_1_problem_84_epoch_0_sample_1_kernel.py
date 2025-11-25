import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int kernel_size,
    int stride,
    int padding,
    int height_out,
    int width_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * out_channels * height_out * width_out) {
        return;
    }

    int w_out = idx % width_out;
    int rem = idx / width_out;
    int h_out = rem % height_out;
    rem /= height_out;
    int c = rem % out_channels;
    rem /= out_channels;
    int n = rem;

    float acc = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride - padding + kh;
            int w_in = w_out * stride - padding + kw;
            
            if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                int input_offset = 
                    n * in_channels * height_in * width_in +
                    c * height_in * width_in +
                    h_in * width_in + 
                    w_in;

                int weight_offset = 
                    c * kernel_size * kernel_size +
                    kh * kernel_size +
                    kw;

                acc += input[input_offset] * weight[weight_offset];
            }
        }
    }

    int output_offset = 
        n * out_channels * height_out * width_out +
        c * height_out * width_out +
        h_out * width_out + 
        w_out;

    output[output_offset] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding
) {
    if (input.dim() != 4) {
        AT_ERROR("Input tensor must be 4-dimensional");
    }
    if (weight.dim() != 4) {
        AT_ERROR("Weight tensor must be 4-dimensional");
    }
    if (weight.size(1) != 1) {
        AT_ERROR("Weight must have 1 input channel per group");
    }
    if (weight.size(2) != kernel_size || weight.size(3) != kernel_size) {
        AT_ERROR("Weight must have kernel_size x kernel_size dimensions");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int out_channels = weight.size(0);

    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, input.options());

    int threads_per_block = 256;
    int total_elements = batch_size * out_channels * height_out * width_out;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_size,
        stride,
        padding,
        height_out,
        width_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding
);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cuda_sources=depthwise_conv_source,
    cpp_sources=depthwise_conv_cpp_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = depthwise_conv.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.kernel_size,
            self.stride,
            self.padding
        )

        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1)

        return output