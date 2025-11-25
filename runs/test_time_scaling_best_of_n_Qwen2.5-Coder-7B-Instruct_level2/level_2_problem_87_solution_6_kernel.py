import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution, subtraction, and Mish activation
conv_sub_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish(float x) {
    return x * tanh(log(1 + exp(x)));
}

__global__ void conv_sub_mish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int channels_in, int channels_out,
    int height_in, int width_in, int kernel_size, int padding, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels_out * height_in * width_in) {
        return;
    }

    int b = idx / (channels_out * height_in * width_in);
    int c_out = (idx % (channels_out * height_in * width_in)) / (height_in * width_in);
    int h_out = (idx % (channels_out * height_in * width_in)) % (height_in * width_in);
    int w_out = (idx % (channels_out * height_in * width_in)) / (height_in * width_in);

    int h_in = h_out * stride - padding;
    int w_in = w_out * stride - padding;

    float sum = 0.0f;
    for (int c_in = 0; c_in < channels_in; ++c_in) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int i_h = h_in + k_h;
                int i_w = w_in + k_w;
                if (i_h >= 0 && i_h < height_in && i_w >= 0 && i_w < width_in) {
                    sum += input[b * channels_in * height_in * width_in + c_in * height_in * width_in + i_h * width_in + i_w] *
                           weight[c_out * channels_in * kernel_size * kernel_size + c_in * kernel_size * kernel_size + k_h * kernel_size + k_w];
                }
            }
        }
    }

    sum -= bias[c_out];
    sum = mish(sum);
    output[idx] = sum;
}

torch::Tensor conv_sub_mish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int padding, int stride
) {
    auto batch_size = input.size(0);
    auto channels_in = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto channels_out = weight.size(0);
    auto height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    auto width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels_out * height_out * width_out + block_size - 1) / block_size;

    conv_sub_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, channels_in, channels_out,
        height_in, width_in, kernel_size, padding, stride
    );

    return output;
}
"""

conv_sub_mish_cpp_source = (
    "torch::Tensor conv_sub_mish_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int padding, int stride);"
)

# Compile the inline CUDA code for convolution, subtraction, and Mish activation
conv_sub_mish = load_inline(
    name="conv_sub_mish",
    cpp_sources=conv_sub_mish_cpp_source,
    cuda_sources=conv_sub_mish_source,
    functions=["conv_sub_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        x = conv_sub_mish.conv_sub_mish_cuda(x, self.conv_weight, self.conv_bias, kernel_size, 1, 1)
        x = x - self.subtract_value_1
        x = x - self.subtract_value_2
        return x


# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 8
    out_channels = 64
    height, width = 256, 256
    kernel_size = 3
    subtract_value_1 = 0.5
    subtract_value_2 = 0.2

    model_new = ModelNew(in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2).cuda()
    inputs = get_inputs()[0].cuda()

    output = model_new(inputs)
    print(output.shape)