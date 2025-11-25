import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv1d_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv1d_transpose_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int o = idx % output_length;
    int c_out = (idx / output_length) % out_channels;
    int b = idx / (output_length * out_channels);

    scalar_t sum = 0;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k = 0; k < kernel_size; ++k) {
            // Correct formula derived after extensive analysis
            int numerator = o + padding - dilation * (kernel_size - 1 - k);
            if (numerator % stride != 0) continue;
            int i = numerator / stride;
            if (i < 0 || i >= input_length) continue;

            scalar_t w_val = weight[c_in][c_out][kernel_size - 1 - k]; // Kernel reversal for transpose
            scalar_t in_val = input[b][c_in][i];

            sum += w_val * in_val;
        }
    }

    output[b][c_out][o] = sum;
}

torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    int out_channels
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_length = input.size(2);

    auto output_length = (input_length - 1) * stride + dilation * (kernel_size - 1) - 2 * padding + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    dim3 threads(256);
    dim3 blocks((batch_size * out_channels * output_length + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_transpose_cuda", ([&] {
        conv1d_transpose_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            input_length,
            output_length,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}
"""

conv1d_transpose_cpp = """
torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    int out_channels
);
"""

conv1d_transpose_module = load_inline(
    name="conv1d_transpose",
    cpp_sources=[conv1d_transpose_cpp],
    cuda_sources=[conv1d_transpose_source],
    functions=["conv1d_transpose_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_transpose_module.conv1d_transpose_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size,
            self.out_channels
        )

def get_inputs():
    batch_size = 16
    in_channels = 32
    length = 131072
    x = torch.randn(batch_size, in_channels, length).cuda()
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]  # in_channels, out_channels, kernel_size, stride, padding, dilation