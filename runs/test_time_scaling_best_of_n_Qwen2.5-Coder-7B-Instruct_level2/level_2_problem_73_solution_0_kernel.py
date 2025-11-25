import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || c_out >= out_channels) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                int h = blockIdx.z * blockDim.z + threadIdx.z;
                int w = blockIdx.w * blockDim.w + threadIdx.w;

                if (h >= height || w >= width) {
                    continue;
                }

                int input_idx = ((n * in_channels + c_in) * height + h) * width + w;
                int weight_idx = ((c_out * in_channels + c_in) * kernel_size + i) * kernel_size + j;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int output_idx = (n * out_channels + c_out) * height * width + blockIdx.z * blockDim.z * width + blockIdx.w * blockDim.w + threadIdx.z * width + threadIdx.w;
    output[output_idx] = sum;
}

torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 threads_per_block(8, 8, 8);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y,
                          (out_channels + threads_per_block.z - 1) / threads_per_block.z,
                          (batch_size + threads_per_block.w - 1) / threads_per_block.w);

    convolution_2d_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv.convolution_2d_cuda(x, self.weight, kernel_size)
        x = self.bn(x)
        x = x * self.scaling_factor
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]