import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for 1D convolution
conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* x, const float* weight, const float* bias, bool has_bias,
                             float* out,
                             int batch_size, int in_channels, int out_channels,
                             int kernel_size, int length, int output_length,
                             int stride, int padding, int dilation, int groups) {
    int c_out = blockIdx.x;
    int n = blockIdx.y;
    int i_block_start = blockIdx.z * blockDim.x;

    int in_per_group = in_channels / groups;
    int out_per_group = out_channels / groups;
    int g = c_out / out_per_group;
    int start_c_in = g * in_per_group;
    int end_c_in = (g + 1) * in_per_group;

    for (int tid = threadIdx.x; tid < blockDim.x; tid += blockDim.x) {
        int i = i_block_start + tid;
        if (i >= output_length) continue;

        float sum = 0.0f;

        int start_input = i * stride - padding;
        for (int c_in = start_c_in; c_in < end_c_in; ++c_in) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_idx = start_input + k * dilation;
                if (input_idx < 0 || input_idx >= length) continue;

                int w_offset = c_out * in_per_group * kernel_size +
                               (c_in - start_c_in) * kernel_size + k;
                int x_offset = n * in_channels * length + c_in * length + input_idx;

                sum += weight[w_offset] * x[x_offset];
            }
        }

        if (has_bias) {
            sum += bias[c_out];
        }

        int out_offset = n * out_channels * output_length + c_out * output_length + i;
        out[out_offset] = sum;
    }
}

torch::Tensor conv1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                            bool has_bias, int kernel_size, int stride, int padding, int dilation, int groups) {
    x = x.contiguous();
    weight = weight.contiguous();
    if (has_bias) bias = bias.contiguous();

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int out_channels = weight.size(0);
    int length = x.size(2);

    int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    int output_length = (length + 2 * padding - effective_kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, x.options());

    int threads_per_block = 128;
    int blocks_per_i = (output_length + threads_per_block - 1) / threads_per_block;
    dim3 grid(out_channels, batch_size, blocks_per_i);
    dim3 block(threads_per_block, 1, 1);

    const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;

    conv1d_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        has_bias,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_size, length, output_length,
        stride, padding, dilation, groups
    );

    return output;
}
"""

# Compile the CUDA kernel
conv1d_cuda = load_inline(
    name="conv1d_cuda",
    cuda_sources=conv1d_source,
    functions=["conv1d_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias

        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_bias = self.bias is not None
        return conv1d_cuda.conv1d_forward(
            x, self.weight, self.bias if has_bias else torch.empty(0),
            has_bias, self.kernel_size, self.stride,
            self.padding, self.dilation, self.groups
        )