import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 2D convolution
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_height, int kernel_width, int stride_h, int stride_w, int padding_h, int padding_w) {
    int batch_id = blockIdx.z;
    int out_ch_id = blockIdx.y;
    int out_row_id = blockIdx.x / stride_w;
    int out_col_id = blockIdx.x % stride_w;

    if (batch_id >= batch_size || out_ch_id >= out_channels || out_row_id >= height_out || out_col_id >= width_out) {
        return;
    }

    float sum = 0.0f;
    int in_row_start = out_row_id * stride_h - padding_h;
    int in_col_start = out_col_id * stride_w - padding_w;

    for (int i = 0; i < kernel_height; ++i) {
        for (int j = 0; j < kernel_width; ++j) {
            int in_row_idx = in_row_start + i;
            int in_col_idx = in_col_start + j;

            if (in_row_idx >= 0 && in_row_idx < height_in && in_col_idx >= 0 && in_col_idx < width_in) {
                for (int k = 0; k < in_channels; ++k) {
                    sum += input[batch_id * in_channels * height_in * width_in + k * height_in * width_in + in_row_idx * width_in + in_col_idx] *
                           weight[out_ch_id * in_channels * kernel_height * kernel_width + k * kernel_height * kernel_width + i * kernel_width + j];
                }
            }
        }
    }

    output[batch_id * out_channels * height_out * width_out + out_ch_id * height_out * width_out + out_row_id * width_out + out_col_id] = sum;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);
    auto height_out = ((height_in - 1) * stride_h - 2 * padding_h + kernel_height) / stride_h + 1;
    auto width_out = ((width_in - 1) * stride_w - 2 * padding_w + kernel_width) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_threads_per_block = block_size;
    const int num_blocks_x = (width_out * stride_w + block_size - 1) / block_size;
    const int num_blocks_y = (out_channels + block_size - 1) / block_size;
    const int num_blocks_z = (batch_size + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<dim3(num_blocks_x, num_blocks_y, num_blocks_z), dim3(num_threads_per_block)>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w
    );

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w);"
)

# Compile the inline CUDA code for transposed 2D convolution
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0)):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d.conv_transpose2d_cuda(x, self.weight, stride=self.stride[0], stride=self.stride[1], padding=self.padding[0], padding=self.padding[1])


# Initialize weights and other parameters
in_channels = 64
out_channels = 128
kernel_size = (3, 5)
stride = (1, 1)
padding = (0, 0)

model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding)
model_new.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
model_new.bias = nn.Parameter(torch.randn(out_channels))

# Test code
def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]