import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int channels, int input_height, int input_width,
    int kernel_size, int stride, int padding,
    int output_height, int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) {
        return;
    }

    int batch = idx / (channels * output_height * output_width);
    int remainder = idx % (channels * output_height * output_width);
    int channel = remainder / (output_height * output_width);
    remainder = remainder % (output_height * output_width);
    int oh = remainder / output_width;
    int ow = remainder % output_width;

    int ih = -padding + oh * stride;
    int iw = -padding + ow * stride;

    scalar_t sum = 0;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            if (ih + ky >= 0 && ih + ky < input_height &&
                iw + kx >= 0 && iw + kx < input_width) {
                sum += input[batch][channel][ih + ky][iw + kx] *
                       weight[channel][0][ky][kx];
            }
        }
    }
    output[batch][channel][oh][ow] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                    int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_size = weight.size(2); // Assume square kernel

    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, 
                              input.options());

    const int num_output_elements = batch_size * channels * output_height * output_width;
    const int threads_per_block = 256;
    const int num_blocks = (num_output_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, channels, input_height, input_width,
            kernel_size, stride, padding,
            output_height, output_width);
    }));

    return output;
}
"""

depthwise_conv2d_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
)

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = depthwise_conv2d(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output