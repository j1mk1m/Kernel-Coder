import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_height, int kernel_width, int stride_h, int stride_w, int pad_h, int pad_w) {
    int batch_idx = blockIdx.z;
    int out_channel_idx = blockIdx.y;
    int out_row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int out_col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row_idx >= height_out || out_col_idx >= width_out) {
        return;
    }

    float sum = 0.0f;
    int in_row_start = max(out_row_idx * stride_h - pad_h, 0);
    int in_row_end = min((out_row_idx + 1) * stride_h - pad_h, height_in);
    int in_col_start = max(out_col_idx * stride_w - pad_w, 0);
    int in_col_end = min((out_col_idx + 1) * stride_w - pad_w, width_in);

    for (int i = in_row_start; i < in_row_end; ++i) {
        for (int j = in_col_start; j < in_col_end; ++j) {
            for (int k = 0; k < in_channels; ++k) {
                int in_index = batch_idx * in_channels * height_in * width_in + k * height_in * width_in + i * width_in + j;
                int weight_index = out_channel_idx * in_channels * kernel_height * kernel_width + k * kernel_height * kernel_width + (in_row_end - i - 1) * kernel_width + (in_col_end - j - 1);
                sum += input[in_index] * weight[weight_index];
            }
        }
    }

    int out_index = batch_idx * out_channels * height_out * width_out + out_channel_idx * height_out * width_out + out_row_idx * width_out + out_col_idx;
    output[out_index] = sum;
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int pad_h, int pad_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto height_out = ((height_in - 1) * stride_h - 2 * pad_h + kernel_height) / stride_h + 1;
    auto width_out = ((width_in - 1) * stride_w - 2 * pad_w + kernel_width) / stride_w + 1;
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const int block_size_x = 16;
    const int block_size_y = 16;
    const int grid_size_x = (width_out + block_size_x - 1) / block_size_x;
    const int grid_size_y = (height_out + block_size_y - 1) / block_size_y;
    const int grid_size_z = batch_size * out_channels;

    transposed_convolution_kernel<<<grid_size_x, block_size_x, 0, at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w);

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int pad_h, int pad_w);"
)

# Compile the inline CUDA code for transposed convolution
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.transposed_conv = transposed_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.transposed_conv.transposed_convolution_cuda(x, self.weight, stride[0], stride[1], padding[0], padding[1])