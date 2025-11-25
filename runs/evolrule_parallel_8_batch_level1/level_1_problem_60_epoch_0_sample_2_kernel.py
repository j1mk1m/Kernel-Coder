import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D convolution
conv3d_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int64_t batch_size, int64_t out_channels, int64_t in_channels,
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t kernel_depth, int64_t kernel_height, int64_t kernel_width,
    int64_t stride, int64_t padding, int64_t dilation) {
    CUDA_1D_KERNEL_LOOP(index, batch_size * out_channels * output_depth * output_height * output_width) {
        int64_t w = index % output_width;
        int64_t h = (index / output_width) % output_height;
        int64_t d = (index / (output_width * output_height)) % output_depth;
        int64_t oc = (index / (output_width * output_height * output_depth)) % out_channels;
        int64_t n = index / (out_channels * output_depth * output_height * output_width);

        scalar_t val = 0;
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t kd = 0; kd < kernel_depth; ++kd) {
                for (int64_t kh = 0; kh < kernel_height; ++kh) {
                    for (int64_t kw = 0; kw < kernel_width; ++kw) {
                        // Compute input coordinates with dilation and padding
                        int64_t id = d * stride - padding + kd * dilation;
                        int64_t ih = h * stride - padding + kh * dilation;
                        int64_t iw = w * stride - padding + kw * dilation;

                        // Check boundaries
                        if (id >= 0 && id < input_depth &&
                            ih >= 0 && ih < input_height &&
                            iw >= 0 && iw < input_width) {
                            val += input[n][ic][id][ih][iw] * weight[oc][ic][kd][kh][kw];
                        }
                    }
                }
            }
        }
        output[n][oc][d][h][w] = val;
    }
}

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight,
    int stride, int padding, int dilation) {
    // Compute output dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_depth = weight.size(2);
    auto kernel_height = weight.size(3);
    auto kernel_width = weight.size(4);

    auto output_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
    auto output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    auto output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    torch::Tensor output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int threads = 256;
    int64_t total = batch_size * out_channels * output_depth * output_height * output_width;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            batch_size, out_channels, in_channels,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            stride, padding, dilation);
    }));

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight,
    int stride, int padding, int dilation);
"""

conv3d_module = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_kernel,
    functions=["conv3d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights similar to PyTorch's Conv3d
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = conv3d_module.conv3d_forward(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

# Ensure that initialization uses the same parameters as original Model
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]