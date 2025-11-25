import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel code for 1D convolution
conv1d_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int length,
    int out_length,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_length) return;

    int batch = idx / (out_channels * out_length);
    int oc = (idx / out_length) % out_channels;
    int op = idx % out_length;

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int pos_in = op * stride + k * dilation - padding;
            if (pos_in < 0 || pos_in >= length) continue;

            const scalar_t in_val = input[batch * in_channels * length + ic * length + pos_in];
            const scalar_t w_val = weight[oc * in_channels * kernel_size + ic * kernel_size + k];
            sum += in_val * w_val;
        }
    }
    output[idx] = sum;
}

torch::Tensor conv1d_forward_cuda(torch::Tensor input, torch::Tensor weight,
                                  int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    int effective_filter_size = dilation * (kernel_size - 1) + 1;
    int out_length = (length + 2 * padding - effective_filter_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_length}, input.options());

    const int threads_per_block = 256;
    const int total_elements = batch_size * out_channels * out_length;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_forward_cuda", ([&] {
        conv1d_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            length,
            out_length,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}
"""

cpp_source = """
torch::Tensor conv1d_forward_cuda(torch::Tensor input, torch::Tensor weight,
                                  int stride, int padding, int dilation);
"""

conv1d_forward = load_inline(
    name="conv1d_forward",
    cpp_sources=cpp_source,
    cuda_sources=conv1d_forward_source,
    functions=["conv1d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == 1, "Groups are not supported in this implementation"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv1d_forward.conv1d_forward_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output