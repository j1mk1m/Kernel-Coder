import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
transposed_conv_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int D, int H, int W, int kD, int kH, int kW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_channels * D * H * W) {
        int b = idx / (out_channels * D * H * W);
        int c_out = (idx % (out_channels * D * H * W)) / (D * H * W);
        int d_out = (idx % (D * H * W)) / (H * W);
        int h_out = (idx % (H * W)) / W;
        int w_out = idx % W;

        float sum = 0.0f;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int dd = 0; dd < kD; ++dd) {
                for (int hh = 0; hh < kH; ++hh) {
                    for (int ww = 0; ww < kW; ++ww) {
                        int d_in = d_out + dd - kD / 2;
                        int h_in = h_out + hh - kH / 2;
                        int w_in = w_out + ww - kW / 2;
                        if (d_in >= 0 && d_in < D && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            int input_idx = ((b * in_channels + c_in) * D + d_in) * H * W + h_in * W + w_in;
                            int weight_idx = ((c_out * in_channels + c_in) * kD + dd) * kH * kW + hh * kW + ww;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor transposed_conv_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto kD = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);

    auto output = torch::zeros({batch_size, out_channels, D, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * D * H * W + block_size - 1) / block_size;

    transposed_conv_3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, D, H, W, kD, kH, kW);

    return output;
}
"""

transposed_conv_3d_cpp_source = (
    "torch::Tensor transposed_conv_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for transposed 3D convolution
transposed_conv_3d = load_inline(
    name="transposed_conv_3d",
    cpp_sources=transposed_conv_3d_cpp_source,
    cuda_sources=transposed_conv_3d_source,
    functions=["transposed_conv_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(input[idx], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for ReLU activation
relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for group normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int channels, int spatial_size, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * spatial_size) {
        int b = idx / (channels * spatial_size);
        int c = (idx % (channels * spatial_size)) / spatial_size;
        int s = idx % spatial_size;

        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int g = 0; g < groups; ++g) {
            if (c / groups == g) {
                int g_start = c * spatial_size;
                int g_end = (c + 1) * spatial_size;
                for (int i = g_start; i < g_end; ++i) {
                    int input_idx = ((b * channels + i) * spatial_size + s);
                    sum += input[input_idx];
                    sum_sq += input[input_idx] * input[input_idx];
                }
            }
        }

        int g_start = c * spatial_size;
        int g_end = (c + 1) * spatial_size;
        for (int i = g_start; i < g_end; ++i) {
            int input_idx = ((b * channels + i) * spatial_size + s);
            output[input_idx] = (input[input_idx] - sum / spatial_size) / sqrt(var[c] + 1e-5);
        }
    }
}

torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto spatial_size = input.size(2);
    auto groups = mean.size(1);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;

    group_norm_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, spatial_size, groups);

    return output;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var);"
)

# Compile the inline CUDA code for group normalization
group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.transposed_conv_3d = transposed_conv_3d
        self.relu = relu
        self.group_norm = group_norm

    def forward(self, x):
        x = self.transposed_conv_3d.transposed_conv_3d_cuda(x)
        x = self.relu.relu_cuda(x)
        x = self.group_norm.group_norm_cuda(x)
        return x