import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose1d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int output_padding) {

    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels * output_length) {
        int batch = output_index / (out_channels * output_length);
        int oc = (output_index / output_length) % out_channels;
        int ol = output_index % output_length;

        scalar_t val = 0;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int k = 0; k < kernel_size; k++) {
                // Compute input position based on transpose conv formula
                int il = (ol + padding - k) / stride;
                if ((ol + padding - k) % stride != 0 || il < 0 || il >= input_length) {
                    continue;
                }
                // Apply output_padding adjustment
                il = (ol + padding - k) / stride - output_padding;
                if (il < 0 || il >= input_length) continue;

                val += weight[oc * in_channels * kernel_size + ic * kernel_size + k] *
                       input[batch * in_channels * input_length + ic * input_length + il];
            }
        }
        output[output_index] = val;
    }
}

torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int threads = 256;
    int elements = batch_size * out_channels * output_length;
    int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose1d_forward", ([&]{
        conv_transpose1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            padding,
            output_padding);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose1d_cpp = """
torch::Tensor conv_transpose1d_forward(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding);
"""

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and bias using PyTorch's default initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Custom CUDA implementation
        output = conv_transpose1d.conv_transpose1d_forward(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output