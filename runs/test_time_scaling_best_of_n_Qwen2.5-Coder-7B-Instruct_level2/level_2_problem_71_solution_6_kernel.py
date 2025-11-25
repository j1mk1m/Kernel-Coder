import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution, division, and LeakyReLU
conv_div_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define DIVISOR 2.0f

__global__ void conv_div_leakyrelu_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int c = idx / (height * width);
    int h = (idx / width) % height;
    int w = idx % width;

    // Convolution
    float sum = 0.0f;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int ih = h + ky;
            int iw = w + kx;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                sum += input[(c * height * width) + (ih * width) + iw];
            }
        }
    }

    // Division
    float result = sum / DIVISOR;

    // LeakyReLU
    output[idx] = result > 0 ? result : result * 0.01f;
}

torch::Tensor conv_div_leakyrelu_cuda(torch::Tensor input, int kernel_size) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height * width + block_size - 1) / block_size;

    conv_div_leakyrelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, kernel_size);

    return output;
}
"""

conv_div_leakyrelu_cpp_source = (
    "torch::Tensor conv_div_leakyrelu_cuda(torch::Tensor input, int kernel_size);"
)

# Compile the inline CUDA code for convolution, division, and LeakyReLU
conv_div_leakyrelu = load_inline(
    name="conv_div_leakyrelu",
    cpp_sources=conv_div_leakyrelu_cpp_source,
    cuda_sources=conv_div_leakyrelu_source,
    functions=["conv_div_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv_div_leakyrelu = conv_div_leakyrelu

    def forward(self, x):
        return self.conv_div_leakyrelu.conv_div_leakyrelu_cuda(x, kernel_size)

# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 8
    out_channels = 64
    height, width = 128, 128
    kernel_size = 3
    divisor = 2

    model = ModelNew(in_channels, out_channels, kernel_size, divisor)
    inputs = get_inputs()
    outputs = model(inputs[0])
    print(outputs.shape)