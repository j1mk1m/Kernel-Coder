import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* weight, const float* bias, float* output,
                             int batch_size, int in_channels, int out_channels, int input_length,
                             int kernel_size, int stride, int padding, int dilation,
                             int output_length, int groups) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_length) return;

    int batch = index / (out_channels * output_length);
    int remainder = index % (out_channels * output_length);
    int out_channel = remainder / output_length;
    int out_pos = remainder % output_length;

    int group = out_channel / (out_channels / groups);
    int group_out_channel = out_channel % (out_channels / groups);

    float sum = 0.0f;
    for (int in_channel_group = 0; in_channel_group < in_channels / groups; ++in_channel_group) {
        int in_channel = group * (in_channels / groups) + in_channel_group;

        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = out_pos * stride - padding + dilation * k;
            if (input_pos >= 0 && input_pos < input_length) {
                int base_weight_index = group * (out_channels / groups) * (in_channels / groups) * kernel_size;
                int out_offset = group_out_channel * (in_channels / groups) * kernel_size;
                int in_offset = in_channel_group * kernel_size;
                int kernel_offset = k;
                int weight_index = base_weight_index + out_offset + in_offset + kernel_offset;

                float w = weight[weight_index];
                float in_val = input[batch * in_channels * input_length + in_channel * input_length + input_pos];
                sum += w * in_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[out_channel];
    }

    output[batch * out_channels * output_length + out_channel * output_length + out_pos] = sum;
}

torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         int stride, int padding, int dilation, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, out_channels, output_length}, options);

    int total_threads = batch_size * out_channels * output_length;
    const int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_cuda", ([&] {
        conv1d_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            (bias.defined()) ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels, input_length,
            kernel_size, stride, padding, dilation,
            output_length, groups
        );
    }));

    return output;
}
"""

conv1d_header = """
torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         int stride, int padding, int dilation, int groups);
"""

conv1d_module = load_inline(
    name="conv1d",
    cpp_sources=conv1d_header,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_module.conv1d_cuda(
            x, self.weight, self.bias if self.bias is not None else torch.Tensor(),
            self.stride, self.padding, self.dilation, self.groups
        )