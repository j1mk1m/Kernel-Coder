import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D convolution
conv1d_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int input_length,
    int out_channels, int kernel_size, int stride, int dilation) {

    int B = blockIdx.z;
    int O = blockIdx.x * blockDim.x + threadIdx.x;
    int K = blockIdx.y * blockDim.y + threadIdx.y;

    if (O >= output.size(1) || K >= kernel_size) return;

    scalar_t value = 0;
    for (int C = 0; C < in_channels; ++C) {
        int I = O * stride + K * dilation;
        if (I < input_length) {
            value += input[B][C][I] * weight[K][C][O];
        }
    }
    atomicAdd(&output[B][O][K], value);
}

torch::Tensor conv1d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int dilation) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto out_channels = weight.size(1);
    auto kernel_size = weight.size(0);

    // Calculate output length
    auto output_length = (input_length - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Output tensor initialization
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    dim3 threads(16, 4);
    dim3 blocks(output_length, kernel_size, batch_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_forward_cuda", ([&] {
        conv1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits>(),
            batch_size, in_channels, input_length,
            out_channels, kernel_size, stride, dilation);
    }));

    return output;
}
"""

conv1d_cpp_source = """
torch::Tensor conv1d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride, int dilation);
"""

conv1d_module = load_inline(
    name="custom_conv1d",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_kernel,
    functions=["conv1d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size, in_channels, out_channels))  # [kernel_size, in_channels, out_channels]
        self.stride = stride
        self.dilation = dilation
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        output = conv1d_module.conv1d_forward_cuda(x, self.weight, self.stride, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
        return output