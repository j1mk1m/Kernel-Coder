import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void convolution_kernel(float* input, float* weight, float* bias, float* output, int channels_in, int channels_out, int height, int width, int kernel_size) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int k = blockIdx.z;

    float sum = 0.0f;

    if (row < height && col < width && k < channels_in) {
        tile[threadIdx.y][threadIdx.x] = input[row * width * channels_in + col * channels_in + k];
        __syncthreads();

        for (int m = 0; m < kernel_size; ++m) {
            for (int n = 0; n < kernel_size; ++n) {
                if (row + m < height && col + n < width) {
                    sum += tile[m][n] * weight[k * kernel_size * kernel_size + m * kernel_size + n];
                }
            }
        }

        __syncthreads();
    }

    if (k == 0) {
        if (row < height && col < width) {
            output[row * width * channels_out + col * channels_out] = sum + bias[0];
        }
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding) {
    int channels_in = input.size(1);
    int channels_out = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int kernel_size = weight.size(2);

    int padded_height = height + 2 * padding;
    int padded_width = width + 2 * padding;

    torch::Tensor padded_input = torch.zeros({batch_size, channels_in, padded_height, padded_width}, device=input.device);
    torch::Tensor output = torch.zeros({batch_size, channels_out, height, width}, device=input.device);

    padded_input.narrow(2, padding, height).narrow(3, padding, width).copy_(input);

    dim3 grid((padded_width + TILE_WIDTH - 1) / TILE_WIDTH, (padded_height + TILE_WIDTH - 1) / TILE_WIDTH, channels_in);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);

    convolution_kernel<<<grid, block>>>(padded_input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), channels_in, channels_out, padded_height, padded_width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.register_buffer('weight', torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer('bias', torch.randn(out_channels))
        self.register_buffer('stride', torch.tensor(stride))
        self.register_buffer('padding', torch.tensor(padding))

    def forward(self, x):
        x = self.conv(x, self.weight, self.bias, stride=int(self.stride.item()), padding=int(self.padding.item()))
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x

# Initialize parameters for ModelNew
in_channels = 64
out_channels = 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

model_new = ModelNew(in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool)

# Example usage
input_tensor = torch.rand(128, 64, 128, 128)
output_tensor = model_new(input_tensor)
print(output_tensor.shape)