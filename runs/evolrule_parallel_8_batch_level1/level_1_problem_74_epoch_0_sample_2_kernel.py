import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose1d_forward_kernel(
    const scalar_t *input,
    const scalar_t *weight,
    scalar_t *output,
    int batch_size,
    int in_channels,
    int input_length,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_length) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length)
        return;

    int ol = idx % output_length;
    int oc = (idx / output_length) % out_channels;
    int n = idx / (out_channels * output_length);

    scalar_t sum = 0;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int temp = ol + padding - dilation * k;
            if (temp % stride != 0)
                continue;
            int il = temp / stride;
            if (il < 0 || il >= input_length)
                continue;

            int input_offset = n * in_channels * input_length + ic * input_length + il;
            int weight_offset = oc * in_channels * kernel_size + ic * kernel_size + k;
            sum += input[input_offset] * weight[weight_offset];
        }
    }

    int output_offset = n * out_channels * output_length + oc * output_length + ol;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose1d_forward(torch::Tensor input,
                                      torch::Tensor weight,
                                      int stride,
                                      int padding,
                                      int dilation,
                                      int kernel_size,
                                      int in_channels,
                                      int out_channels,
                                      int input_length) {
    int output_length = (input_length - 1)*stride - 2*padding + dilation*(kernel_size -1) +1;

    auto output = torch::empty({input.size(0), out_channels, output_length}, input.options());

    const int threads_per_block = 256;
    int total_elements = input.size(0)*out_channels*output_length;
    int blocks_per_grid = (total_elements + threads_per_block -1)/threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose1d_forward", ([&] {
        conv_transpose1d_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.size(0),
            in_channels,
            input_length,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_length);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor conv_transpose1d_forward(torch::Tensor input,
                                      torch::Tensor weight,
                                      int stride,
                                      int padding,
                                      int dilation,
                                      int kernel_size,
                                      int in_channels,
                                      int out_channels,
                                      int input_length);
"""

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose1d_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.size(2)
        out = conv_transpose1d.conv_transpose1d_forward(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size,
            self.in_channels,
            self.out_channels,
            input_length
        )
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            out += bias_view
        return out