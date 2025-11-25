import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_size,
    int stride, int padding, int dilation,
    int output_height, int output_width) {

    const int batch_size = input.size(0);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Each thread computes one output element
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * output_height * output_width) return;

    int batch = out_idx / (output_height * output_width);
    int rem = out_idx % (output_height * output_width);
    int oh = rem / output_width;
    int ow = rem % output_width;

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Compute input coordinates
                int ih = oh - padding - kh * dilation;
                int iw = ow - padding - kw * dilation;
                if (ih % stride == 0 && iw % stride == 0) {
                    ih /= stride;
                    iw /= stride;
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        for (int oc = 0; oc < out_channels; oc++) {
                            // Transposed convolution uses weight[oc][ic][kh][kw]
                            sum += input[batch][ic][ih][iw] * weight[oc][ic][kh][kw];
                        }
                    }
                }
            }
        }
    }
    output[batch][oc][oh][ow] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride, int padding, int dilation) {

    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    const auto batch_size = input.size(0);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    // Compute output dimensions
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + 2 * padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_forward", ([&] {
        conv_transpose2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_size,
            stride, padding, dilation,
            output_height, output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   int stride, int padding, int dilation);
"""

# Compile the CUDA kernel
conv_transpose2d = load_inline(
    name='conv_transpose2d',
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose_source,
    functions=['conv_transpose2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Bias is not used in this example as per the original model's default
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):
        return conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )