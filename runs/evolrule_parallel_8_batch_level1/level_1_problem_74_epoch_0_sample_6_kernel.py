import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose1d_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation,
    bool has_bias,
    const scalar_t* __restrict__ bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int l_out = idx % output_length;
    int c_out = (idx / output_length) % out_channels;
    int n = idx / (output_length * out_channels);

    scalar_t sum = 0.0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k = 0; k < kernel_size; ++k) {
            int l_in = (l_out - k * dilation + padding) / stride;
            if (l_in >= 0 && l_in < input_length) {
                int w_idx = c_in * out_channels * kernel_size + c_out * kernel_size + k;
                scalar_t w = weight[w_idx];
                int in_idx = n * in_channels * input_length + c_in * input_length + l_in;
                sum += input[in_idx] * w;
            }
        }
    }

    if (has_bias) {
        sum += bias[c_out];
    }

    int out_idx = n * out_channels * output_length + c_out * output_length + l_out;
    output[out_idx] = sum;
}

at::Tensor conv_transpose1d_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    const int input_length = input.size(2);
    const int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    at::Tensor output = at::empty({batch_size, out_channels, output_length}, input.options());

    const int total_threads = batch_size * out_channels * output_length;
    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose1d_forward_cuda", ([&] {
        conv_transpose1d_forward<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            padding,
            dilation,
            has_bias,
            bias.data<scalar_t>()
        );
    }));

    return output;
}
"""

conv_transpose1d_cpp_source = """
at::Tensor conv_transpose1d_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
"""

# Load the CUDA extension
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cuda_sources=conv_transpose1d_source,
    cpp_sources=conv_transpose1d_cpp_source,
    functions=["conv_transpose1d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_tensor = self.bias if self.bias_flag else torch.empty(0, device=x.device, dtype=x.dtype)
        return conv_transpose1d.conv_transpose1d_forward_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride,
            self.padding,
            self.dilation,
            self.bias_flag
        )