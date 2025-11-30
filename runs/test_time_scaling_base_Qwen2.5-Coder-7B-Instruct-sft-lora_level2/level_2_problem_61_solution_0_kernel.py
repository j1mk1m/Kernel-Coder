import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
transposed_conv_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv_3d_kernel(float* input, float* weight, float* output, int batch_size, int in_channels, int out_channels, int D, int H, int W, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * D * H * W) return;

    int o_d = idx / (H * W * out_channels);
    int o_h = (idx % (H * W * out_channels)) / (W * out_channels);
    int o_w = ((idx % (H * W * out_channels)) % (W * out_channels)) / out_channels;
    int c_in = idx % out_channels;

    float sum = 0.0f;
    for (int d = 0; d < kernel_size; ++d) {
        for (int h = 0; h < kernel_size; ++h) {
            for (int w = 0; w < kernel_size; ++w) {
                int i_d = o_d * stride - padding + d;
                int i_h = o_h * stride - padding + h;
                int i_w = o_w * stride - padding + w;
                if (i_d >= 0 && i_d < D && i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                    int i_idx = (o_d * D + i_d) * (H * W) + (o_h * H + i_h) * W + o_w * W + i_w;
                    int w_idx = (c_in * kernel_size + d) * (kernel_size * kernel_size) + (h * kernel_size + w);
                    sum += input[i_idx * in_channels + c_in] * weight[w_idx];
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor transposed_conv_3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, D, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * D * H * W + block_size - 1) / block_size;

    transposed_conv_3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, D, H, W, kernel_size, stride, padding);

    return output;
}
"""

transposed_conv_3d_cpp_source = (
    "torch::Tensor transposed_conv_3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
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

# Define the custom CUDA kernel for ReLU
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = max(input[idx], 0.0f);
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

# Compile the inline CUDA code for ReLU
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

__global__ void group_norm_kernel(float* input, float* mean, float* var, float* gamma, float* beta, float* output, int batch_size, int channels, int spatial_size, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * spatial_size) return;

    int g = idx / (channels * spatial_size);
    int c_in = (idx % (channels * spatial_size)) / spatial_size;
    int s_idx = idx % spatial_size;

    float sum = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        sum += input[(g * channels + c_in) * spatial_size + i];
    }
    mean[g * channels + c_in] = sum / spatial_size;

    float var_sum = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        var_sum += pow(input[(g * channels + c_in) * spatial_size + i] - mean[g * channels + c_in], 2.0f);
    }
    var[g * channels + c_in] = var_sum / spatial_size;

    output[idx] = gamma[g * channels + c_in] * (input[idx] - mean[g * channels + c_in]) / sqrt(var[g * channels + c_in] + 1e-5f) + beta[g * channels + c_in];
}

torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int groups) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto spatial_size = input.size(2) * input.size(3) * input.size(4);
    auto spatial_size_per_group = spatial_size / groups;

    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({groups, channels}, input.options());
    auto var = torch::zeros({groups, channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;

    group_norm_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, spatial_size, groups);

    return output;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int groups);"
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
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]