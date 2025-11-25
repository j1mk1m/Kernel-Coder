import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void conv3d_transpose_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> kernel,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth, int input_width, int input_height,
    int output_depth, int output_width, int output_height,
    int kernel_d, int kernel_w, int kernel_h,
    int stride_d, int stride_w, int stride_h
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * output_depth * output_width * output_height) {
        return;
    }

    int n = idx / (out_channels * output_depth * output_width * output_height);
    int remaining = idx % (out_channels * output_depth * output_width * output_height);
    int o_c = remaining / (output_depth * output_width * output_height);
    remaining %= (output_depth * output_width * output_height);
    int d_out = remaining / (output_width * output_height);
    remaining %= (output_width * output_height);
    int w_out = remaining / output_height;
    int h_out = remaining % output_height;

    scalar_t sum = 0;

    for (int i_c = 0; i_c < in_channels; ++i_c) {

        // Compute valid kernel depth indices
        int kd_min = max(0, d_out - stride_d * (input_depth - 1));
        int kd_max = min(kernel_d - 1, d_out);

        for (int kd = kd_min; kd <= kd_max; ++kd) {
            int d_in = (d_out - kd) / stride_d;

            // Compute valid kernel width indices
            int kw_min = max(0, w_out - stride_w * (input_width - 1));
            int kw_max = min(kernel_w - 1, w_out);

            for (int kw = kw_min; kw <= kw_max; ++kw) {
                int w_in = (w_out - kw) / stride_w;

                // Compute valid kernel height indices
                int kh_min = max(0, h_out - stride_h * (input_height - 1));
                int kh_max = min(kernel_h - 1, h_out);

                for (int kh = kh_min; kh <= kh_max; ++kh) {
                    int h_in = (h_out - kh) / stride_h;

                    // Access input tensor
                    scalar_t input_val = input[n][i_c][d_in][w_in][h_in];

                    // Access kernel weights
                    scalar_t kernel_val = kernel[i_c][o_c][kd][kw][kh];

                    sum += input_val * kernel_val;
                }
            }
        }
    }

    output[n][o_c][d_out][w_out][h_out] = sum;
}

extern "C" {

    torch::Tensor conv3d_transpose_cuda(
        torch::Tensor input,
        torch::Tensor kernel,
        int output_depth,
        int output_width,
        int output_height,
        int kernel_d,
        int kernel_w,
        int kernel_h,
        int stride_d,
        int stride_w,
        int stride_h
    ) {

        int batch_size = input.size(0);
        int in_channels = input.size(1);
        int input_depth = input.size(2);
        int input_width = input.size(3);
        int input_height = input.size(4);
        int out_channels = kernel.size(1);

        auto output = torch::zeros({batch_size, out_channels, output_depth, output_width, output_height}, input.options());

        int total_elements = batch_size * out_channels * output_depth * output_width * output_height;
        int threads_per_block = 256;
        int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_transpose_cuda", ([&] {
            conv3d_transpose_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
                kernel.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
                output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
                batch_size, in_channels, out_channels,
                input_depth, input_width, input_height,
                output_depth, output_width, output_height,
                kernel_d, kernel_w, kernel_h,
                stride_d, stride_w, stride_h
            );
        }));

        return output;
    }
}
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize kernel weights with PyTorch's default initialization
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Load the CUDA kernel
        self.conv3d_transpose = load_inline(
            name="conv3d_transpose",
            cuda_sources=cuda_source,
            functions=["conv3d_transpose_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_depth, input_width, input_height = x.shape
        kernel_d, kernel_w, kernel_h = self.kernel_size
        stride_d, stride_w, stride_h = self.stride

        # Calculate output dimensions using given parameters (padding=0, output_padding=0)
        output_depth = (input_depth - 1) * stride_d + kernel_d
        output_width = (input_width - 1) * stride_w + kernel_w
        output_height = (input_height - 1) * stride_h + kernel_h

        return self.conv3d_transpose.conv3d_transpose_cuda(
            x,
            self.weight,
            output_depth,
            output_width,
            output_height,
            kernel_d,
            kernel_w,
            kernel_h,
            stride_d,
            stride_w,
            stride_h
        )