import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose3d
convtranspose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convtranspose3d_forward_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding, int dilation) {
    int b = blockIdx.z;
    int c_out = blockIdx.y;
    int d_out = blockIdx.x / (height_out * width_out);
    int h_out = (blockIdx.x % (height_out * width_out)) / width_out;
    int w_out = blockIdx.x % width_out;

    int d_in_start = max(d_out * stride - padding, 0);
    int d_in_end = min(d_in_start + kernel_size * dilation, depth_in);
    int h_in_start = max(h_out * stride - padding, 0);
    int h_in_end = min(h_in_start + kernel_size * dilation, height_in);
    int w_in_start = max(w_out * stride - padding, 0);
    int w_in_end = min(w_in_start + kernel_size * dilation, width_in);

    float sum = 0.0f;
    for (int d_in = d_in_start; d_in < d_in_end; ++d_in) {
        for (int h_in = h_in_start; h_in < h_in_end; ++h_in) {
            for (int w_in = w_in_start; w_in < w_in_end; ++w_in) {
                int i = ((b * in_channels + c_out) * depth_in + d_in) * height_in * width_in +
                        (h_in * width_in + w_in);
                int j = ((c_out * kernel_size + d_in - d_in_start) * kernel_size + h_in - h_in_start) *
                        kernel_size + w_in - w_in_start;
                sum += input[i] * weight[j];
            }
        }
    }
    int o = ((b * out_channels + c_out) * depth_out + d_out) * height_out * width_out +
            (h_out * width_out + w_out);
    output[o] = sum;
}

torch::Tensor convtranspose3d_forward_cuda(torch::Tensor input, torch::Tensor weight) {
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
    auto stride = 2; // Assuming fixed stride for simplicity
    auto padding = 1; // Assuming fixed padding for simplicity
    auto dilation = 1; // Assuming fixed dilation for simplicity

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (depth_out * height_out * width_out + block_size - 1) / block_size;

    dim3 grid_dim(num_blocks, out_channels, batch_size);
    dim3 block_dim(block_size);

    convtranspose3d_forward_kernel<<<grid_dim, block_dim>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding, dilation);

    return output;
}
"""

convtranspose3d_cpp_source = (
    "torch::Tensor convtranspose3d_forward_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for ConvTranspose3d
convtranspose3d = load_inline(
    name="convtranspose3d",
    cpp_sources=convtranspose3d_cpp_source,
    cuda_sources=convtranspose3d_source,
    functions=["convtranspose3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for MaxPool3d
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool3d_forward_kernel(const float* input, float* output, int batch_size, int in_channels, int depth_in, int height_in, int width_in, int kernel_size, int stride, int padding) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int d_out = blockIdx.x / (height_out * width_out);
    int h_out = (blockIdx.x % (height_out * width_out)) / width_out;
    int w_out = blockIdx.x % width_out;

    int d_in_start = max(d_out * stride - padding, 0);
    int d_in_end = min(d_in_start + kernel_size, depth_in);
    int h_in_start = max(h_out * stride - padding, 0);
    int h_in_end = min(h_in_start + kernel_size, height_in);
    int w_in_start = max(w_out * stride - padding, 0);
    int w_in_end = min(w_in_start + kernel_size, width_in);

    float max_val = -std::numeric_limits<float>::infinity();
    for (int d_in = d_in_start; d_in < d_in_end; ++d_in) {
        for (int h_in = h_in_start; h_in < h_in_end; ++h_in) {
            for (int w_in = w_in_start; w_in < w_in_end; ++w_in) {
                int i = ((b * in_channels + c) * depth_in + d_in) * height_in * width_in +
                        (h_in * width_in + w_in);
                max_val = std::max(max_val, input[i]);
            }
        }
    }
    int o = ((b * in_channels + c) * depth_out + d_out) * height_out * width_out +
            (h_out * width_out + w_out);
    output[o] = max_val;
}

torch::Tensor maxpool3d_forward_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto kernel_size = 2; // Assuming fixed kernel size for simplicity
    auto stride = 2; // Assuming fixed stride for simplicity
    auto padding = 0; // Assuming fixed padding for simplicity

    auto depth_out = (depth_in + padding * 2 - kernel_size) / stride + 1;
    auto height_out = (height_in + padding * 2 - kernel_size) / stride + 1;
    auto width_out = (width_in + padding * 2 - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (depth_out * height_out * width_out + block_size - 1) / block_size;

    dim3 grid_dim(num_blocks, in_channels, batch_size);
    dim3 block_dim(block_size);

    maxpool3d_forward_kernel<<<grid_dim, block_dim>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, depth_in, height_in, width_in, kernel_size, stride, padding);

    return output;
}
"""

maxpool3d_cpp_source = (
    "torch::Tensor maxpool3d_forward_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for MaxPool3d
maxpool3d = load_inline(
    name="maxpool3d",
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_source,
    functions=["maxpool3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_forward_kernel(const float* input, float* output, int batch_size, int in_channels, int depth, int height, int width) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int d = blockIdx.x / (height * width);
    int h = (blockIdx.x % (height * width)) / width;
    int w = blockIdx.x % width;

    int i = ((b * in_channels + c) * depth + d) * height * width +
            (h * width + w);
    float exp_val = exp(input[i]);
    float sum_exp = 0.0f;
    for (int i = 0; i < depth * height * width; ++i) {
        sum_exp += exp(input[b * in_channels * depth * height * width + c * depth * height * width + i]);
    }
    output[i] = exp_val / sum_exp;
}

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (depth * height * width + block_size - 1) / block_size;

    dim3 grid_dim(num_blocks, in_channels, batch_size);
    dim3 block_dim(block_size);

    softmax_forward_kernel<<<grid_dim, block_dim>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, depth, height, width);

    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_forward_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = convtranspose3d
        self.max_pool = maxpool3d
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose.convtranspose3d_forward_cuda(x, self.weight)
        x = self.max_pool.maxpool3d_forward_cuda(x)
        x = softmax.softmax_forward_cuda(x)
        x = x - self.subtract.view(1, -1, 1, 1, 1)
        x = torch.sigmoid(x) * x
        x = torch.max(x, dim=1)[0]
        return x