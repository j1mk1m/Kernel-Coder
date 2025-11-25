import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height, int width, int out_channels, int kernel_size) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z;
    int w = blockIdx.w;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int ih = h * kernel_size + i;
            int iw = w * kernel_size + j;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                sum += input[n * in_channels * height * width + c * height * width + ih * width + iw] * weight[c * kernel_size * kernel_size + i * kernel_size + j];
            }
        }
    }

    output[n * out_channels * height * width + c * height * width + h * width + w] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y,
                          (out_channels + threads_per_block.z - 1) / threads_per_block.z);

    conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height, width, out_channels, kernel_size);

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 2D convolution
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv2d = conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d.conv2d_cuda(x, self.weight)

# Initialize weights
weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()

# Create an instance of the model
model_new = ModelNew(in_channels, out_channels, kernel_size)

# Get inputs
inputs = get_inputs()
x = inputs[0].cuda()

# Forward pass
output = model_new.forward(x)
print(output.shape)  # Should print torch.Size([8, 128, 512, 1024])