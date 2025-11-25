import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel implementation for 1D convolution
conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_forward_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int length,
    int out_channels, int kernel_size, int stride, int dilation,
    int output_length
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_length)
        return;

    int batch = index / (out_channels * output_length);
    int remainder = index % (out_channels * output_length);
    int oc = remainder / output_length;
    int l = remainder % output_length;

    int start = l * stride;
    float sum = 0.0f;

    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = start + k * dilation;
        if (input_pos < 0 || input_pos >= length)
            continue;

        for (int ic = 0; ic < in_channels; ++ic) {
            int input_offset = batch * in_channels * length + ic * length + input_pos;
            float in_val = input[input_offset];

            int weight_offset = oc * in_channels * kernel_size + ic * kernel_size + k;
            float wt_val = weight[weight_offset];

            sum += in_val * wt_val;
        }
    }

    int output_offset = batch * out_channels * output_length + oc * output_length + l;
    output[output_offset] = sum;
}

torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight, int stride, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int output_length = (length - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads_per_block = 256;
    const int total_elements = batch_size * out_channels * output_length;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_forward", ([&] {
        conv1d_forward_kernel<<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, in_channels, length,
            out_channels, kernel_size, stride, dilation,
            output_length
        );
    }));

    return output;
}
"""

conv1d_header = """
torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight, int stride, int dilation);
"""

# Python wrapper for the CUDA kernel
conv1d = load_inline(
    name="conv1d",
    cpp_sources=conv1d_header,
    cuda_sources=conv1d_source,
    functions=["conv1d_forward"],
    verbose=True,
)

# New Model class using the custom kernel
class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, dilation=1, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        # Initialize weights like PyTorch's Conv1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.stride = stride
        self.dilation = dilation
        self.conv1d_forward = conv1d.conv1d_forward

    def forward(self, x):
        output = self.conv1d_forward(x, self.weight, self.stride, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output