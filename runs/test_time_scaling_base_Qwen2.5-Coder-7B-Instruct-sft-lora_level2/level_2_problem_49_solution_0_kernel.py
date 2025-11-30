import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(float* input, float* weight, float* output, int batch_size, int in_channels, int out_channels, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int output_padding_d, int output_padding_h, int output_padding_w) {
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= D_out || h >= H_out || w >= W_out) return;

    float sum = 0.0f;
    for (int i = 0; i < D_in; ++i) {
        for (int j = 0; j < H_in; ++j) {
            for (int k = 0; k < W_in; ++k) {
                int in_idx = ((d * stride_d + i - padding_d) * H_in + j - padding_h) * W_in + k - padding_w;
                int weight_idx = (((D_in - i - 1) * stride_d + d - padding_d) * H_in + h - padding_h) * W_in + w - padding_w;
                int out_idx = ((d * stride_d + i - padding_d) * H_in + j - padding_h) * W_in + k - padding_w;
                if (in_idx >= 0 && in_idx < D_in * H_in * W_in && weight_idx >= 0 && weight_idx < D_in * H_in * W_in && out_idx >= 0 && out_idx < D_out * H_out * W_out) {
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    int out_idx = ((d * stride_d + i - padding_d) * H_in + j - padding_h) * W_in + k - padding_w;
    if (out_idx >= 0 && out_idx < D_out * H_out * W_out) {
        output[out_idx] = sum;
    }
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int output_padding_d, int output_padding_h, int output_padding_w) {
    auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());

    const dim3 threads_per_block(16, 16, 1);
    const dim3 blocks_per_grid((W_out + threads_per_block.x - 1) / threads_per_block.x, (H_out + threads_per_block.y - 1) / threads_per_block.y, (D_out + threads_per_block.z - 1) / threads_per_block.z);

    conv_transpose_3d_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, D_in, H_in, W_in, D_out, H_out, W_out, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w);

    return output;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int output_padding_d, int output_padding_h, int output_padding_w);"
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

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(float* input, float* output, int batch_size, int channels, int D, int H, int W) {
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= channels || d >= D || h >= H) return;

    float max_val = -INFINITY;
    for (int i = 0; i < W; ++i) {
        int idx = ((c * D + d) * H + h) * W + i;
        if (input[idx] > max_val) {
            max_val = input[idx];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < W; ++i) {
        int idx = ((c * D + d) * H + h) * W + i;
        sum_exp += exp(input[idx] - max_val);
    }

    int out_idx = ((c * D + d) * H + h) * W + i;
    output[out_idx] = exp(input[out_idx] - max_val) / sum_exp;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);

    const dim3 threads_per_block(16, 16, 1);
    const dim3 blocks_per_grid((W + threads_per_block.x - 1) / threads_per_block.x, (H + threads_per_block.y - 1) / threads_per_block.y, (channels + threads_per_block.z - 1) / threads_per_block.z);

    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1), input.size(2), input.size(3), input.size(4));

    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for Sigmoid
sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel(float* input, float* output, int batch_size, int channels, int D, int H, int W) {
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= channels || d >= D || h >= H) return;

    int idx = ((c * D + d) * H + h) * W;
    output[idx] = 1.0f / (1.0f + exp(-input[idx]));
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);

    const dim3 threads_per_block(16, 16, 1);
    const dim3 blocks_per_grid((W + threads_per_block.x - 1) / threads_per_block.x, (H + threads_per_block.y - 1) / threads_per_block.y, (channels + threads_per_block.z - 1) / threads_per_block.z);

    sigmoid_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1), input.size(2), input.size(3), input.size(4));

    return output;
}
"""

sigmoid_cpp_source = (
    "torch::Tensor sigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Sigmoid
sigmoid = load_inline(
    name="sigmoid",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.softmax = softmax
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_cuda(x, ..., batch_size=batch_size, in_channels=in_channels, out_channels=out_channels, D_in=D, H_in=H, W_in=W, D_out=D_out, H_out=H_out, W_out=W_out, stride_d=stride, stride_h=stride, stride_w=stride, padding_d=padding, padding_h=padding, padding_w=padding, output_padding_d=output_padding, output_padding_h=output_padding, output_padding_w=output_padding)
        x = self.softmax.softmax_cuda(x)
        x = self.sigmoid.sigmoid_cuda(x)
        return x