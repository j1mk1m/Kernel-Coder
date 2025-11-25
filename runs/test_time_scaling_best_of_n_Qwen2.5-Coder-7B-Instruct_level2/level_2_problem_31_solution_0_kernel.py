import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int b = blockIdx.z;
    int o = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (b >= batch_size || o >= out_channels || i >= in_channels * (width - kernel_size + 1)) return;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        for (int c = 0; c < in_channels; ++c) {
            int h_idx = i / in_channels + k;
            int w_idx = i % in_channels;
            sum += input[b * in_channels * height * width + c * height * width + h_idx * width + w_idx] * weight[o * in_channels * kernel_size + c * kernel_size + k];
        }
    }

    output[b * out_channels * height * width + o * height * width + i] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 blocks_per_grid((in_channels * (width - kernel_size + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE, out_channels / BLOCK_SIZE, batch_size);

    convolution_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size);"
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
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = convolution.convolution_cuda(x, self.conv_weight, self.bias, kernel_size)
        x = torch.min(x, torch.tensor(self.constant_value))
        x = x + self.bias
        x = x * self.scaling_factor
        return x

# Initialize the model and inputs
model_new = ModelNew(*get_init_inputs())
inputs = get_inputs()

# Forward pass through the model
output = model_new(inputs[0])
print(output.shape)