import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

custom_conv1d_source = """
__global__ void custom_conv1d(
    const float* input,
    float* output,
    const float* weight,
    int batch,
    int in_channels,
    int out_channels,
    int length,
    int kernel_size,
    int stride,
    int dilation,
    int output_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * output_length) return;

    int o = idx % output_length;
    int oc = (idx / output_length) % out_channels;
    int b = (idx / output_length) / out_channels;

    int s = o * stride;
    float sum = 0.0;

    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = s + k * dilation;
        if (input_pos >= length) continue;
        for (int ic = 0; ic < in_channels; ++ic) {
            int input_offset = b * in_channels * length + ic * length + input_pos;
            int weight_offset = oc * in_channels * kernel_size + ic * kernel_size + k;
            sum += input[input_offset] * weight[weight_offset];
        }
    }

    int output_offset = b * out_channels * output_length + oc * output_length + o;
    output[output_offset] = sum;
}

torch::Tensor custom_conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int dilation
) {
    int kernel_size = weight.size(2);
    int in_channels = weight.size(1);
    int out_channels = weight.size(0);
    int batch = input.size(0);
    int length = input.size(2);

    int effective_kernel_width = (kernel_size - 1)*dilation + 1;
    int output_length = (length - effective_kernel_width) / stride + 1;

    auto output = torch::empty({batch, out_channels, output_length}, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch * out_channels * output_length;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    custom_conv1d<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        length,
        kernel_size,
        stride,
        dilation,
        output_length
    );

    cudaDeviceSynchronize();
    return output;
}
"""

custom_conv1d_cpp_source = """
#include <torch/extension.h>

torch::Tensor custom_conv1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int dilation);
"""

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv1d_cpp_source,
    cuda_sources=custom_conv1d_source,
    functions=["custom_conv1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = custom_conv.custom_conv1d_cuda(x, self.weight, self.stride, self.dilation)
        if self.bias and self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1)
        return output