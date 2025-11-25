import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int j = idx % output_length;
    int oc = (idx / output_length) % out_channels;
    int b = idx / (output_length * out_channels);

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int temp_j = j - k*dilation;
            if (temp_j % stride != 0) continue;
            int i = temp_j / stride;
            if (i < 0 || i >= input_length) continue;

            int kernel_offset = ic * out_channels * kernel_size + oc * kernel_size + k;
            float w = kernel[kernel_offset];
            int input_offset = b * in_channels * input_length + ic * input_length + i;
            sum += w * input[input_offset];
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }

    int output_offset = b * out_channels * output_length + oc * output_length + j;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = kernel.size(1);
    int kernel_size = kernel.size(2);

    int output_length = (input_length - 1)*stride - 2*padding + dilation*(kernel_size - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    int threads_per_block = 256;
    int blocks_per_grid = (batch_size * out_channels * output_length + threads_per_block -1) / threads_per_block;

    conv_transpose1d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias
    );

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed");
    }

    return output;
}
"""

conv_transpose1d_header = """
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
"""

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_header,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.empty((in_channels, out_channels, kernel_size)))
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

        self.conv_transpose1d = conv_transpose1d

    def forward(self, x):
        has_bias = self.bias is not None
        return self.conv_transpose1d.conv_transpose1d_cuda(
            x,
            self.weight,
            self.bias if has_bias else torch.empty(0),
            self.stride,
            self.padding,
            self.dilation,
            has_bias
        )