import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int channels, int height, int width, int kernel_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height || col >= width) {
        return;
    }

    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_row = row + ky - kernel_size / 2;
                int in_col = col + kx - kernel_size / 2;
                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                    sum += input[(in_row * width + in_col) * channels + c] * weight[ky * kernel_size + kx];
                }
            }
        }
    }
    output[row * width + col] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({channels, height, width}, input.options());

    const int block_size = 16;
    dim3 grid((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    convolution_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), channels, height, width, kernel_size);

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
        x = torch.relu(x)
        x = x * torch.clamp((x + 3) / 6, 0, 1)
        return x

# Initialize the model with weights
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def init_model(in_channels, out_channels, kernel_size):
    model = ModelNew(in_channels, out_channels, kernel_size)
    model.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    return model

# Get inputs
inputs = get_inputs()

# Initialize model
model = init_model(in_channels, out_channels, kernel_size)

# Forward pass
output = model(inputs[0])
print(output.shape)