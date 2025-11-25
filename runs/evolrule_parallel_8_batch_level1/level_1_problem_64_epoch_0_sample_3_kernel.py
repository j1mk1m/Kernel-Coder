import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int input_length,
    const int output_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length)
        return;

    int n = idx / (out_channels * output_length);
    int remaining = idx % (out_channels * output_length);
    int out_c = remaining / output_length;
    int out_pos = remaining % output_length;

    int out_channels_per_group = out_channels / groups;
    int group_id = out_c / out_channels_per_group;
    int out_c_in_group = out_c % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;
    int in_c_base = group_id * in_channels_per_group;

    float acc = 0.0f;

    for (int in_c = 0; in_c < in_channels_per_group; ++in_c) {
        int in_c_global = in_c_base + in_c;

        for (int kernel_pos = 0; kernel_pos < kernel_size; ++kernel_pos) {
            int in_pos = (out_pos - kernel_pos - padding) / stride;
            if (in_pos < 0 || in_pos >= input_length)
                continue;

            int kernel_index = in_c_global * out_channels_per_group * kernel_size +
                               out_c_in_group * kernel_size +
                               kernel_pos;

            float w = kernel[kernel_index];
            int input_offset = n * in_channels * input_length +
                               in_c_global * input_length +
                               in_pos;
            float in_val = input[input_offset];

            acc += w * in_val;
        }
    }

    int output_offset = n * out_channels * output_length +
                        out_c * output_length +
                        out_pos;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input,
                                   torch::Tensor kernel,
                                   int batch_size,
                                   int in_channels,
                                   int out_channels,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int groups,
                                   int input_length,
                                   int output_length) {
    input = input.contiguous();
    kernel = kernel.contiguous();

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    int block_size = 256;
    int grid_size = (batch_size * out_channels * output_length + block_size - 1) / block_size;

    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);

    conv_transpose1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        input_length,
        output_length
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return torch::Tensor(); // Return invalid tensor to trigger error
    }
    cudaDeviceSynchronize();

    return output;
}
}
"""

conv_transpose1d_cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv_transpose1d_cuda(torch::Tensor input,
                                   torch::Tensor kernel,
                                   int batch_size,
                                   int in_channels,
                                   int out_channels,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int groups,
                                   int input_length,
                                   int output_length);
"""

# Compile the CUDA code
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights with the same distribution as PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_length = x.size()
        output_length = (input_length - 1) * self.stride - 2 * self.padding + \
                        self.kernel_size + self.output_padding

        output = conv_transpose1d.conv_transpose1d_cuda(
            x,
            self.weight,
            batch_size,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            input_length,
            output_length
        )
        return output