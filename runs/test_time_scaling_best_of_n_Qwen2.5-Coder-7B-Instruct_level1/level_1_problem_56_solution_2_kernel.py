import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for conv2d
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to perform the convolution
__global__ void conv2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height_in, int width_in, int out_channels, int height_weight, int width_weight, int stride_h, int stride_w, int padding_h, int padding_w) {
    int batch_id = blockIdx.z;
    int out_channel_id = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height_in || col >= width_in) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < height_weight; ++i) {
        for (int j = 0; j < width_weight; ++j) {
            int in_row = row + i - padding_h;
            int in_col = col + j - padding_w;
            if (in_row >= 0 && in_row < height_in && in_col >= 0 && in_col < width_in) {
                int in_idx = (batch_id * in_channels + 0) * height_in * width_in + (in_row * width_in + in_col);
                int w_idx = (out_channel_id * in_channels + 0) * height_weight * width_weight + (i * width_weight + j);
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    int out_idx = (batch_id * out_channels + out_channel_id) * height_in * width_in + (row * width_in + col);
    output[out_idx] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto out_channels = weight.size(0);
    auto height_weight = weight.size(2);
    auto width_weight = weight.size(3);

    auto output = torch::zeros({batch_size, out_channels, height_in, width_in}, input.options());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((width_in + threads_per_block.x - 1) / threads_per_block.x, (height_in + threads_per_block.y - 1) / threads_per_block.y, batch_size * out_channels);

    conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height_in, width_in, out_channels, height_weight, width_weight, stride_h, stride_w, padding_h, padding_w);

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride_h, int stride_w, int padding_h, int padding_w);"
)

# Compile the inline CUDA code for conv2d
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d.conv2d_cuda(x, self.weight, stride=self.stride[0], stride=self.stride[1], padding=self.padding[0], padding=self.padding[1])