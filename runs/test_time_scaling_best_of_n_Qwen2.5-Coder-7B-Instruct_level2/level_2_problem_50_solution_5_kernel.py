import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int b = idx / (out_channels * depth * height * width);
    int o = (idx % (out_channels * depth * height * width)) / (depth * height * width);
    int d_out = (idx % (out_channels * depth * height * width)) / (height * width);
    int h_out = (idx % (out_channels * depth * height * width)) / width;
    int w_out = idx % width;

    int d_in = d_out * stride - padding;
    int h_in = h_out * stride - padding;
    int w_in = w_out * stride - padding;

    float sum = 0.0f;
    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int d_in_k = d_in + k_d;
                int h_in_k = h_in + k_h;
                int w_in_k = w_in + k_w;

                if (d_in_k >= 0 && d_in_k < depth && h_in_k >= 0 && h_in_k < height && w_in_k >= 0 && w_in_k < width) {
                    int i_idx = b * in_channels * depth * height * width + (d_in_k * height * width + h_in_k * width + w_in_k) * in_channels;
                    int w_idx = o * kernel_size * kernel_size * kernel_size * in_channels + (k_d * kernel_size * kernel_size + k_h * kernel_size + k_w) * in_channels;
                    sum += input[i_idx] * weight[w_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);
    auto stride = 2;
    auto padding = 1;

    auto output = torch::zeros({batch_size, out_channels, depth, height, width});

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth * height * width + block_size - 1) / block_size;

    conv_transpose_3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding);

    return output;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for bias addition
bias_addition_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bias_addition_kernel(const float* input, const float* bias, float* output, int batch_size, int out_channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int b = idx / (out_channels * depth * height * width);
    int o = (idx % (out_channels * depth * height * width)) / (depth * height * width);
    int d = (idx % (out_channels * depth * height * width)) / (height * width);
    int h = (idx % (out_channels * depth * height * width)) / width;
    int w = idx % width;

    output[idx] = input[idx] + bias[o];
}

torch::Tensor bias_addition_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto out_channels = bias.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth * height * width + block_size - 1) / block_size;

    bias_addition_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, depth, height, width);

    return output;
}
"""

bias_addition_cpp_source = (
    "torch::Tensor bias_addition_cuda(torch::Tensor input, torch::Tensor bias);"
)

# Compile the inline CUDA code for bias addition
bias_addition = load_inline(
    name="bias_addition",
    cpp_sources=bias_addition_cpp_source,
    cuda_sources=bias_addition_source,
    functions=["bias_addition_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_cuda(x, self.weight)
        x = x * self.scale1
        x = self.avg_pool(x)
        x = bias_addition.bias_addition_cuda(x, self.bias)
        x = x * self.scale2
        return x

# Example usage
if __name__ == "__main__":
    model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)