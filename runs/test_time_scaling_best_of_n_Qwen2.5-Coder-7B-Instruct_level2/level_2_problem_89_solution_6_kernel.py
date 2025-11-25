import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution transpose
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding) {
    int b = blockIdx.x / (height_out * width_out);
    int h = (blockIdx.x % (height_out * width_out)) / width_out;
    int w = blockIdx.x % width_out;

    int d_in_start = h * stride - padding;
    int h_in_start = w * stride - padding;
    int w_in_start = b * stride - padding;

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int k_d = 0; k_d < kernel_size; ++k_d) {
                for (int k_h = 0; k_h < kernel_size; ++k_h) {
                    for (int k_w = 0; k_w < kernel_size; ++k_w) {
                        int d_in = d_in_start + k_d;
                        int h_in = h_in_start + k_h;
                        int w_in = w_in_start + k_w;

                        if (d_in >= 0 && d_in < depth_in && h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                            int in_idx = ((b * in_channels + c_in) * depth_in + d_in) * height_in * width_in + h_in * width_in + w_in;
                            int weight_idx = ((c_out * in_channels + c_in) * kernel_size + k_d) * kernel_size * kernel_size + k_h * kernel_size + k_w;
                            sum += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        int out_idx = ((b * out_channels + c_out) * depth_out + h) * width_out + w;
        output[out_idx] = sum;
    }
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto depth_out = weight.size(2);
    auto height_out = weight.size(3);
    auto width_out = weight.size(4);
    auto kernel_size = weight.size(5);
    auto stride = 2; // Assuming stride is always 2
    auto padding = 1; // Assuming padding is always 1

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * height_out * width_out + block_size - 1) / block_size;

    conv_transpose_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding);

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for convolution transpose
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for max pool and softmax fusion
max_pool_and_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool_and_softmax_kernel(const float* input, float* output, int batch_size, int channels, int depth, int height, int width, int pool_depth, int pool_height, int pool_width) {
    int b = blockIdx.x / (channels * depth * height);
    int c = (blockIdx.x % (channels * depth * height)) / (depth * height);
    int d = (blockIdx.x % (depth * height)) / height;
    int h = blockIdx.x % height;

    int d_out = d / pool_depth;
    int h_out = h / pool_height;
    int w_out = b % width;

    float max_val = -FLT_MAX;
    for (int k_d = 0; k_d < pool_depth; ++k_d) {
        for (int k_h = 0; k_h < pool_height; ++k_h) {
            for (int k_w = 0; k_w < pool_width; ++k_w) {
                int d_in = d + k_d;
                int h_in = h + k_h;
                int w_in = w_out + k_w;

                if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int idx = ((b * channels + c) * depth + d_in) * height * width + h_in * width + w_in;
                    max_val = fmax(max_val, input[idx]);
                }
            }
        }
    }

    output[blockIdx.x] = max_val;
}

torch::Tensor max_pool_and_softmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto pool_depth = 2; // Assuming pool size is always 2
    auto pool_height = 2; // Assuming pool size is always 2
    auto pool_width = 2; // Assuming pool size is always 2

    auto output = torch::zeros({batch_size * channels * depth * height * width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * depth * height * width + block_size - 1) / block_size;

    max_pool_and_softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, depth, height, width, pool_depth, pool_height, pool_width);

    return output.reshape({batch_size, channels, depth, height, width});
}
"""

max_pool_and_softmax_cpp_source = (
    "torch::Tensor max_pool_and_softmax_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for max pool and softmax fusion
max_pool_and_softmax = load_inline(
    name="max_pool_and_softmax",
    cpp_sources=max_pool_and_softmax_cpp_source,
    cuda_sources=max_pool_and_softmax_source,
    functions=["max_pool_and_softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.max_pool_and_softmax = max_pool_and_softmax

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_cuda(x, self.weight)
        x = self.max_pool_and_softmax.max_pool_and_softmax_cuda(x)
        x = torch.softmax(x, dim=1)
        x = x - self.subtract.view(1, -1, 1, 1, 1)
        x = torch.sigmoid(x) * x
        x = torch.max(x, dim=1)[0]
        return x

    def weight(self):
        # Placeholder for weight tensor
        return torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size).cuda()

    def subtract(self):
        # Placeholder for subtract parameter
        return torch.randn(out_channels).cuda()


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]