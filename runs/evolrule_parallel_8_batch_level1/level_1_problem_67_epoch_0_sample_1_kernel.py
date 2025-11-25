import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel and wrapper
custom_conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv1d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    int has_bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int input_length,
    int output_length) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_length) return;

    int batch = index / (out_channels * output_length);
    int remainder = index % (out_channels * output_length);
    int oc = remainder / output_length;
    int t_out = remainder % output_length;

    int out_channels_per_group = out_channels / groups;
    int group = oc / out_channels_per_group;
    int in_channels_per_group = in_channels / groups;
    int in_channels_start = group * in_channels_per_group;

    int start = t_out * stride - padding;
    float sum = 0.0f;

    for (int k = 0; k < kernel_size; ++k) {
        int input_time = start + k * dilation;
        if (input_time < 0 || input_time >= input_length) continue;

        for (int ic = in_channels_start; ic < in_channels_start + in_channels_per_group; ++ic) {
            int w_idx = oc * in_channels_per_group * kernel_size + (ic - in_channels_start) * kernel_size + k;
            float w = weight[w_idx];
            int in_idx = batch * in_channels * input_length + ic * input_length + input_time;
            sum += input[in_idx] * w;
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }

    int out_idx = batch * out_channels * output_length + oc * output_length + t_out;
    output[out_idx] = sum;
}

torch::Tensor custom_conv1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int effective_kernel_size = dilation * (kernel_size -1) +1;
    int output_length = (input_length + 2*padding - effective_kernel_size)/stride +1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    int threads_per_block = 256;
    int num_elements = batch_size * out_channels * output_length;
    int blocks_per_grid = (num_elements + threads_per_block -1) / threads_per_block;

    int has_bias = 0;
    if (bias.defined()) {
        if (bias.size(0) == out_channels) {
            has_bias = 1;
        } else {
            throw std::runtime_error("Bias tensor must have size equal to out_channels");
        }
    }

    custom_conv1d_forward<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        has_bias,
        batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
        input_length, output_length
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));

    return output;
}
"""

custom_conv1d_cpp_source = """
torch::Tensor custom_conv1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups);
"""

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv1d_cpp_source,
    cuda_sources=custom_conv1d_source,
    functions=["custom_conv1d"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=False):
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

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, dtype=torch.float32, device=x.device)
        return custom_conv.custom_conv1d(
            x,
            self.weight,
            bias_tensor,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )