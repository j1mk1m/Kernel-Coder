import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int c_in = blockIdx.x;
    int h_out = blockIdx.w / width;
    int w_out = blockIdx.w % width;

    float sum = 0.0f;
    for (int k_h = 0; k_h < kernel_size; ++k_h) {
        for (int k_w = 0; k_w < kernel_size; ++k_w) {
            int i_h = h_out * stride + k_h - pad;
            int i_w = w_out * stride + k_w - pad;
            if (i_h >= 0 && i_h < height && i_w >= 0 && i_w < width) {
                int i_idx = n * in_channels * height * width + c_in * height * width + i_h * width + i_w;
                int w_idx = c_out * kernel_size * kernel_size * in_channels + c_in * kernel_size * kernel_size + k_h * kernel_size + k_w;
                sum += input[i_idx] * weight[w_idx];
            }
        }
    }

    int o_idx = n * out_channels * height * width + c_out * height * width + h_out * width + w_out;
    output[o_idx] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int pad) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, (height - kernel_size + 2 * pad) / stride + 1, (width - kernel_size + 2 * pad) / stride + 1}, input.options());

    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(out_channels, in_channels, batch_size);
    dim3 grid_size((height - kernel_size + 2 * pad) / stride + 1, (width - kernel_size + 2 * pad) / stride + 1);

    convolution_kernel<<<grid_size, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int pad);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for group normalization
group_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_normalization_kernel(const float* input, float* output, float* mean, float* var, int batch_size, int groups, int channels, int height, int width) {
    int g = blockIdx.x;
    int c = blockIdx.y;
    int n = blockIdx.z;
    int h = blockIdx.w / width;
    int w = blockIdx.w % width;

    int g_start = g * channels / groups;
    int g_end = (g + 1) * channels / groups;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int ch = g_start; ch < g_end; ++ch) {
        int i_idx = n * channels * height * width + ch * height * width + h * width + w;
        sum += input[i_idx];
        sum_sq += input[i_idx] * input[i_idx];
    }

    mean[n * groups + g] = sum / (channels / groups * height * width);
    var[n * groups + g] = sum_sq / (channels / groups * height * width) - mean[n * groups + g] * mean[n * groups + g];

    float inv_std = 1.0f / sqrt(var[n * groups + g] + eps);
    for (int ch = g_start; ch < g_end; ++ch) {
        int i_idx = n * channels * height * width + ch * height * width + h * width + w;
        output[i_idx] = (input[i_idx] - mean[n * groups + g]) * inv_std * gamma[g] + beta[g];
    }
}

torch::Tensor group_normalization_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto batch_size = input.size(0);
    auto groups = input.size(1) / gamma.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({batch_size, groups});
    auto var = torch::zeros({batch_size, groups});

    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(groups, channels, batch_size);
    dim3 grid_size(height * width);

    group_normalization_kernel<<<grid_size, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), batch_size, groups, channels, height, width);

    return output;
}
"""

group_normalization_cpp_source = (
    "torch::Tensor group_normalization_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);"
)

# Compile the inline CUDA code for group normalization
group_normalization = load_inline(
    name="group_normalization",
    cpp_sources=group_normalization_cpp_source,
    cuda_sources=group_normalization_source,
    functions=["group_normalization_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.gamma = nn.Parameter(torch.randn(num_groups))
        self.beta = nn.Parameter(torch.randn(num_groups))
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, stride=1, pad=1)
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = group_normalization.group_normalization_cuda(x, self.gamma, self.beta, self.eps)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]