import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void conv_transpose1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int length_in,
    int output_length,
    int in_channels_per_group,
    int out_channels_per_group
) {
    int batch_id = blockIdx.x / out_channels;
    int out_channel = blockIdx.x % out_channels;
    int g = out_channel / out_channels_per_group;
    int out_channel_in_group = out_channel % out_channels_per_group;
    int in_channels_start = g * in_channels_per_group;

    for (int n = threadIdx.x; n < output_length; n += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            int reversed_k = kernel_size - 1 - k;
            int input_pos = (n + padding - reversed_k - output_padding) / stride;
            if (input_pos < 0 || input_pos >= length_in) {
                continue;
            }
            for (int i_c_local = 0; i_c_local < in_channels_per_group; ++i_c_local) {
                int i_c = in_channels_start + i_c_local;
                int w_index = i_c_local * out_channels_per_group * kernel_size +
                              out_channel_in_group * kernel_size +
                              reversed_k;
                float w = weight[w_index];
                int input_offset = batch_id * in_channels * length_in +
                                   i_c * length_in +
                                   input_pos;
                float in_val = input[input_offset];
                acc += in_val * w;
            }
        }
        int output_offset = batch_id * out_channels * output_length +
                            out_channel * output_length +
                            n;
        output[output_offset] = acc;
    }
}

torch::Tensor conv_transpose1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int length_in = input.size(2);
    const int out_channels = weight.size(1) * groups;
    const int kernel_size = weight.size(2);
    const int in_channels_per_group = in_channels / groups;
    const int out_channels_per_group = out_channels / groups;

    const int output_length = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    const int threads_per_block = 256;
    const int blocks_per_group = (output_length + threads_per_block - 1) / threads_per_block;
    const int total_blocks = batch_size * out_channels;

    dim3 grid(total_blocks);
    dim3 block(threads_per_block);

    conv_transpose1d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        length_in,
        output_length,
        in_channels_per_group,
        out_channels_per_group
    );

    return output;
}
"""

conv_transpose1d_cpp_source = (
    "torch::Tensor conv_transpose1d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups);"
)

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels // groups, kernel_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose1d.conv_transpose1d_forward_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )