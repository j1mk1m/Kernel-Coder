import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose_1d_forward(
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
    int dilation
) {
    int batch = blockIdx.x;
    int oc = blockIdx.y;
    int o = threadIdx.x + blockDim.x * blockIdx.z;

    if (batch >= batch_size || oc >= out_channels || o >= output_length) return;

    scalar_t sum = 0.0;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int i = (o - k * dilation) / stride - padding;
            if (i >= 0 && i < input_length) {
                int weight_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
                int input_idx = batch * in_channels * input_length + ic * input_length + i;
                sum += weight[weight_idx] * input[input_idx];
            }
        }
    }

    int output_idx = batch * out_channels * output_length + oc * output_length + o;
    output[output_idx] = sum;
}

std::vector<torch::Tensor> conv_transpose_1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    const int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    const int threads_per_block = 256;
    const int num_blocks_z = (output_length + threads_per_block - 1) / threads_per_block;

    dim3 blocks(batch_size, out_channels, num_blocks_z);
    dim3 threads(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_1d_forward", ([&] {
        conv_transpose_1d_forward<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return {output};
}
"""

conv_transpose_1d_cpp = """
std::vector<torch::Tensor> conv_transpose_1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

conv_transpose_1d = load_inline(
    name="conv_transpose_1d",
    cpp_sources=conv_transpose_1d_cpp,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose_1d_forward_cuda"],
    verbose=True
)

import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = conv_transpose_1d.conv_transpose_1d_forward_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )[0]
        if self.bias is not None:
            outputs = outputs + self.bias.view(1, -1, 1)
        return outputs