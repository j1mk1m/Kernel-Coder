import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_5D_INDEX(b, c, d, h, w, s_b, s_c, s_d, s_h, s_w) \\
    ((b)*s_b + (c)*s_c + (d)*s_d + (h)*s_h + (w)*s_w)

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width) {

    // Flatten spatial coordinates to 1D block index
    int idx = blockIdx.x;
    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int d_out = idx / (output_width * output_height);

    // Thread indices for batch and output channels
    for (int batch = threadIdx.x; batch < batch_size; batch += blockDim.x) {
        for (int c_out = threadIdx.y; c_out < out_channels; c_out += blockDim.y) {
            scalar_t sum = 0.0;

            for (int k_d = 0; k_d < kernel_size; ++k_d) {
                for (int k_h = 0; k_h < kernel_size; ++k_h) {
                    for (int k_w = 0; k_w < kernel_size; ++k_w) {
                        // Compute input indices
                        int d_in = (d_out - k_d + 2 * padding - output_padding) / stride;
                        int h_in = (h_out - k_h + 2 * padding - output_padding) / stride;
                        int w_in = (w_out - k_w + 2 * padding - output_padding) / stride;

                        if (d_in < 0 || d_in >= input_depth) continue;
                        if (h_in < 0 || h_in >= input_height) continue;
                        if (w_in < 0 || w_in >= input_width) continue;

                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            // Weight index: [C_in, C_out, K_d, K_h, K_w]
                            int weight_idx = CUDA_5D_INDEX(
                                c_in, c_out, k_d, k_h, k_w,
                                in_channels * kernel_size * kernel_size * kernel_size,
                                kernel_size * kernel_size * kernel_size,
                                kernel_size * kernel_size,
                                kernel_size,
                                1);

                            // Input index: [Batch, C_in, D_in, H_in, W_in]
                            int input_idx = CUDA_5D_INDEX(
                                batch, c_in, d_in, h_in, w_in,
                                in_channels * input_depth * input_height * input_width,
                                input_depth * input_height * input_width,
                                input_height * input_width,
                                input_width,
                                1);

                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }

            // Output index: [Batch, C_out, D_out, H_out, W_out]
            int out_idx = CUDA_5D_INDEX(
                batch, c_out, d_out, h_out, w_out,
                out_channels * output_depth * output_height * output_width,
                output_depth * output_height * output_width,
                output_height * output_width,
                output_width,
                1);

            atomicAdd(&output[out_idx], sum);
        }
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int output_depth,
    int output_height,
    int output_width) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0); // weight is [in_channels, out_channels, ...]
    const int kernel_size = weight.size(2);

    // Grid and block dimensions
    dim3 threads(32, 8); // Threads handle batches and channels
    int num_blocks = output_depth * output_height * output_width;
    dim3 blocks(num_blocks);

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, in_channels, out_channels, kernel_size, stride, padding, output_padding,
        input.size(2), input.size(3), input.size(4),
        output_depth, output_height, output_width);

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int output_depth,
    int output_height,
    int output_width);
"""

# Compile the CUDA kernel
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cuda_sources=conv_transpose3d_source,
    cpp_sources=conv_transpose3d_cpp_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # He initialization

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        input_depth = x.size(2)
        input_height = x.size(3)
        input_width = x.size(4)

        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.weight.size(2) + self.output_padding
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.weight.size(3) + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.weight.size(4) + self.output_padding

        # Launch CUDA kernel
        output = conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            output_depth,
            output_height,
            output_width
        )

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)

        return output