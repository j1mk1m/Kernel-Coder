import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv_1d_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::DefaultPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::DefaultPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,3,torch::DefaultPtrTraits> output,
    int batch_size, int in_channels, int out_channels, int kernel_size,
    int input_length, int output_length,
    int stride, int padding, int dilation) {

    int b = blockIdx.x / out_channels;
    int o = blockIdx.x % out_channels;

    for (int j = threadIdx.x; j < output_length; j += blockDim.x) {
        scalar_t sum = 0;
        for (int i_c = 0; i_c < in_channels; ++i_c) {
            for (int k = 0; k < kernel_size; ++k) {
                int numerator = j + padding - dilation * k;
                if (numerator % stride != 0)
                    continue;
                int i = numerator / stride;
                if (i >=0 && i < input_length) {
                    sum += input[b][i_c][i] * weight[i_c][o][k];
                }
            }
        }
        output[b][o][j] = sum;
    }
}

torch::Tensor transposed_conv_1d_cuda(torch::Tensor input, torch::Tensor weight,
                                      int stride, int padding, int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    // Compute output_length
    const int output_length = (input_length - 1) * stride - 2 * padding +
                              dilation * (kernel_size - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    const int threads_per_block = 256;
    const int blocks = batch_size * out_channels;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv_1d_cuda", ([&] {
        transposed_conv_1d_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.packed_accessor<scalar_t,3,torch::DefaultPtrTraits>(),
            weight.packed_accessor<scalar_t,3,torch::DefaultPtrTraits>(),
            output.packed_accessor<scalar_t,3,torch::DefaultPtrTraits>(),
            batch_size, in_channels, out_channels, kernel_size,
            input_length, output_length, stride, padding, dilation);
    }));

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_conv_1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_conv_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = transposed_conv.transposed_conv_1d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        return out

# Test code (not part of the required output)
if __name__ == "__main__":
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    length = 131072
    stride = 2
    padding = 1
    dilation = 2

    model = ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation)
    x = torch.randn(batch_size, in_channels, length).cuda()
    y = model(x)
    print(y.shape)