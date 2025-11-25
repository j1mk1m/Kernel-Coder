import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
convolution_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_weight, int width_weight, int stride, int padding) {
    int batch_id = blockIdx.z;
    int channel_id = blockIdx.y * blockDim.y + threadIdx.y;
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id >= batch_size || channel_id >= out_channels || row_id >= height_in) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < in_channels; ++i) {
        for (int j = 0; j < height_weight; ++j) {
            for (int k = 0; k < width_weight; ++k) {
                int input_row = row_id * stride - padding + j;
                int input_col = i * stride - padding + k;
                if (input_row >= 0 && input_row < height_in && input_col >= 0 && input_col < width_in) {
                    int input_index = batch_id * in_channels * height_in * width_in + i * height_in * width_in + input_row * width_in + input_col;
                    int weight_index = channel_id * in_channels * height_weight * width_weight + i * height_weight * width_weight + j * width_weight + k;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
    }
    int output_index = batch_id * out_channels * height_in * width_in + channel_id * height_in * width_in + row_id * width_in;
    output[output_index] = sum;
}

torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto height_weight = weight.size(2);
    auto width_weight = weight.size(3);
    auto stride = 1; // Assuming stride is always 1 for simplicity
    auto padding = 0; // Assuming no padding for simplicity

    auto output = torch::zeros({batch_size, out_channels, height_in, width_in}, input.options());

    const int block_size = 16;
    const int grid_size_x = (height_in + block_size - 1) / block_size;
    const int grid_size_y = (out_channels + block_size - 1) / block_size;
    const int grid_size_z = batch_size;

    convolution_2d_kernel<<<grid_size_x * grid_size_y * grid_size_z, block_size * block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height_in, width_in, height_weight, width_weight, stride, padding);

    return output;
}
"""

convolution_2d_cpp_source = (
    "torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 2D convolution
convolution_2d = load_inline(
    name="convolution_2d",
    cpp_sources=convolution_2d_cpp_source,
    cuda_sources=convolution_2d_source,
    functions=["convolution_2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = convolution_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d.convolution_2d_cuda(x, self.weight)


# Initialize weights
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
model_new = ModelNew(in_channels, out_channels, kernel_size)
model_new.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))