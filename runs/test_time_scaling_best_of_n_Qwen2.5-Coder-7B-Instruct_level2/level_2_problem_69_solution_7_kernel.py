import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int input_height, int input_width, int output_height, int output_width, int channels, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output_height || col >= output_width) {
        return;
    }

    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int in_row = row + i - kernel_size / 2;
                int in_col = col + j - kernel_size / 2;
                if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                    sum += input[(in_row * input_width + in_col) * channels + c] * weight[i * kernel_size + j];
                }
            }
        }
    }
    output[row * output_width + col] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto output_height = input_height - weight.size(2) + 1;
    auto output_width = input_width - weight.size(3) + 1;
    auto channels = input.size(1);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({input.size(0), channels, output_height, output_width}, input.options());

    const int block_size = 16;
    dim3 grid(output_width / block_size, output_height / block_size, 1);
    dim3 block(block_size, block_size, 1);

    convolution_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), input_height, input_width, output_height, output_width, channels, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight);"
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
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight)
        x = torch.nn.functional.hardswish(x)
        x = torch.relu(x)
        return x

# Initialize the model
model_new = ModelNew(in_channels, out_channels, kernel_size)

# Get inputs
inputs = get_inputs()

# Forward pass
output = model_new(inputs[0])
print(output.shape)