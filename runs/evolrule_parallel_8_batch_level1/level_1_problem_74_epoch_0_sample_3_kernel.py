import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_1d_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int out_ch = rem / output_length;
    int i = rem % output_length;

    float total = 0.0;

    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k = 0; k < kernel_size; ++k) {
            int kernel_element_index = kernel_size - 1 - k; // flipped kernel
            int o = (i + padding - kernel_element_index * dilation) / stride;

            if (o < 0 || o >= input_length) continue;

            int weight_idx = in_ch * out_channels * kernel_size + out_ch * kernel_size + kernel_element_index;
            float w = weight[weight_idx];

            int input_idx = b * in_channels * input_length + in_ch * input_length + o;
            float in_val = input[input_idx];

            total += w * in_val;
        }
    }

    int output_idx = b * out_channels * output_length + out_ch * output_length + i;
    output[output_idx] = total;
}

torch::Tensor conv_transpose_1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    int input_length = input.size(2);

    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    int total_threads = batch_size * out_channels * output_length;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    conv_transpose_1d_forward<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose_1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose_1d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv_transpose.conv_transpose_1d_forward_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output