import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int channels, int height, int width, int kernel_size) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    if (h >= height || w >= width) return;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int ih = h + i;
            int iw = w + j;
            if (ih >= height || iw >= width) continue;
            sum += input[(c * height + ih) * width + iw] * weight[i * kernel_size + j];
        }
    }
    output[c * height * width + h * width + w] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size) {
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = torch::zeros({channels, height, width}, input.options());

    const int block_size = 16;
    dim3 grid(channels, (height + block_size - 1) / block_size, (width + block_size - 1) / block_size);
    dim3 block(block_size, block_size, 1);

    convolution_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size);"
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
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, kernel_size)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum along channel dimension
        return x