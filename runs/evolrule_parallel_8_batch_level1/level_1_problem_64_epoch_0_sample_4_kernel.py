import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose1d_cuda_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int input_length,
    int output_length,
    int input_stride_b,
    int input_stride_c,
    int output_stride_b,
    int output_stride_c,
    int weight_stride_0,
    int weight_stride_1,
    int weight_stride_2
) {
    int batch = blockIdx.x;
    int oc = blockIdx.y;
    int t_block = blockIdx.z;
    int t = threadIdx.x + t_block * blockDim.x;

    if (t >= output_length) return;

    scalar_t acc = 0.0;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int numerator = t + output_padding - k;
            if (numerator % stride != 0) continue;
            int input_t = numerator / stride;
            if (input_t < 0 || input_t >= input_length) continue;

            int input_idx = batch * input_stride_b + ic * input_stride_c + input_t;
            int weight_idx = ic * weight_stride_0 + oc * weight_stride_1 + k * weight_stride_2;

            acc += input[input_idx] * weight[weight_idx];
        }
    }

    int output_idx = batch * output_stride_b + oc * output_stride_c + t;
    output[output_idx] = acc;
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int kernel_size) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(1);

    const int output_length = (input_length - 1) * stride + kernel_size - 2 * padding + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    const int input_stride_b = in_channels * input_length;
    const int input_stride_c = input_length;
    const int output_stride_b = out_channels * output_length;
    const int output_stride_c = output_length;
    const int weight_stride_0 = out_channels * kernel_size;
    const int weight_stride_1 = kernel_size;
    const int weight_stride_2 = 1;

    const int threadsPerBlock = 256;
    const int blocksPerBlock_t = (output_length + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blocksPerGrid(batch_size, out_channels, blocksPerBlock_t);
    dim3 threadsPerBlock(threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_cuda_forward<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            input_length,
            output_length,
            input_stride_b,
            input_stride_c,
            output_stride_b,
            output_stride_c,
            weight_stride_0,
            weight_stride_1,
            weight_stride_2
        );
    }));

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));

    return output;
}
"""

conv_transpose1d_header = """
torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int kernel_size);
"""

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_header,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        assert groups == 1, "Groups other than 1 not supported"
        assert not bias, "Bias not supported"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose1d.conv_transpose1d_cuda(
            x, self.weight, self.stride, self.padding, self.output_padding, self.kernel_size
        )