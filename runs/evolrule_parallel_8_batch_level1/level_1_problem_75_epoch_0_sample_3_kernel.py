import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* output,
    int batch_size, int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int input_height, int input_width,
    int output_height, int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (out_channels * output_width * output_height);

    // Determine group
    int out_per_group = out_channels / groups;
    int group_id = c_out / out_per_group;
    int c_out_in_group = c_out % out_per_group;

    int in_channels_per_group = in_channels / groups;
    int in_channel_start = group_id * in_channels_per_group;

    scalar_t sum = 0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            // Compute input coordinates
            int h_in = (h_out - kh*dilation_h - padding_h) / stride_h;
            int w_in = (w_out - kw*dilation_w - padding_w) / stride_w;

            if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width)
                continue;

            for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
                int in_channel = in_channel_start + c_in;

                // Input index
                int in_offset = n * in_channels * input_height * input_width +
                                in_channel * input_height * input_width +
                                h_in * input_width + w_in;
                scalar_t in_val = input[in_offset];

                // Weight index (corrected for PyTorch's weight storage)
                int group_in_channel = group_id * in_channels_per_group + c_in;
                int weight_offset = group_in_channel * out_per_group * kernel_h * kernel_w +
                                    c_out_in_group * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                scalar_t weight_val = weight[weight_offset];

                sum += in_val * weight_val;
            }
        }
    }

    // Output index
    int out_offset = n * out_channels * output_height * output_width +
                     c_out * output_height * output_width +
                     h_out * output_width + w_out;
    output[out_offset] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = weight.size(1) * groups;

    // Compute output size
    int output_height = (input_height - 1) * stride_h - 2*padding_h + dilation_h*(kernel_h-1) + 1;
    int output_width = (input_width - 1) * stride_w - 2*padding_w + dilation_w*(kernel_w-1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    int threads_per_block = 256;
    int num_elements = batch_size * out_channels * output_height * output_width;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups,
            input_height, input_width,
            output_height, output_width
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
);
"""

# Compile the inline CUDA code for ConvTranspose2d
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        # Create the standard PyTorch layer to initialize parameters
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.dilation_h, self.dilation_w = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters and weights from the standard layer
        weight = self.conv_transpose2d.weight
        # Launch the custom CUDA kernel
        return conv_transpose2d.conv_transpose2d_cuda(
            x, weight,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w,
            self.groups
        )