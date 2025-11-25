import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution with ReLU
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_relu_kernel(const float* input, const float* weight, float* output, int batch_size, int channels_in, int channels_out, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels_out * height * width) {
        return;
    }

    int b = idx / (channels_out * height * width);
    int c_out = (idx % (channels_out * height * width)) / (height * width);
    int h = (idx % (channels_out * height * width)) % (height * width);
    int w = ((idx % (channels_out * height * width)) % (height * width)) / kernel_size;

    float sum = 0.0f;
    for (int c_in = 0; c_in < channels_in; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = h + kh;
                int iw = w + kw;
                if (ih >= height || iw >= width) {
                    continue;
                }
                int i_idx = b * channels_in * height * width + c_in * height * width + ih * width + iw;
                int k_idx = c_out * channels_in * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                sum += input[i_idx] * weight[k_idx];
            }
        }
    }

    output[idx] = fmaxf(sum, 0.0f);
}

torch::Tensor conv_relu_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto channels_in = input.size(1);
    auto channels_out = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, channels_out, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels_out * height * width + block_size - 1) / block_size;

    conv_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels_in, channels_out, height, width, kernel_size);

    return output;
}
"""

conv_relu_cpp_source = (
    "torch::Tensor conv_relu_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for convolution with ReLU
conv_relu = load_inline(
    name="conv_relu",
    cpp_sources=conv_relu_cpp_source,
    cuda_sources=conv_relu_source,
    functions=["conv_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_relu = conv_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_relu.conv_relu_cuda(x, self.conv.weight)
        x = x * torch.clamp((x + 3) / 6, 0, 1)
        return x

# Example usage
if __name__ == "__main__":
    model = ModelNew(in_channels, out_channels, kernel_size)
    inputs = get_inputs()
    outputs = model(inputs[0])
    print(outputs.shape)