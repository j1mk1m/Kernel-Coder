import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.z;  // Batch index
    int c_out = blockIdx.y;  // Output channel index
    int c_in = blockIdx.x;  // Input channel index
    int h = threadIdx.y + blockIdx.y * blockDim.y;  // Height index
    int w = threadIdx.x + blockIdx.x * blockDim.x;  // Width index

    float sum = 0.0f;
    if (h >= 0 && h < height && w >= 0 && w < width) {
        for (int k = 0; k < kernel_size; ++k) {
            for (int j = 0; j < kernel_size; ++j) {
                int ih = h - k + kernel_size / 2;
                int iw = w - j + kernel_size / 2;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    sum += input[n * in_channels * height * width + c_in * height * width + ih * width + iw] *
                           weight[c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + k * kernel_size + j];
                }
            }
        }
        output[n * out_channels * height * width + c_out * height * width + h * width + w] = sum;
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 blocks(out_channels, in_channels, batch_size);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    convolution_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

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
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Convolution
        x = convolution.convolution_cuda(x, self.conv_weight)
        # Tanh activation
        x = torch.tanh(x)
        # Scaling
        x = x * self.scaling_factor
        # Bias addition
        x = x + self.bias
        # Max-pooling
        x = self.max_pool(x)
        return x