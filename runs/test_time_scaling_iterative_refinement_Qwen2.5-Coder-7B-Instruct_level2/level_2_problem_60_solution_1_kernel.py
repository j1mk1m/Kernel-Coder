import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || c_out >= out_channels || d_out >= depth_out) return;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_d = 0; k_d < kernel_size; ++k_d) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    int d_in = d_out * stride - padding + k_d;
                    int h_in = (height_out - 1) * stride - padding + k_h;
                    int w_in = (width_out - 1) * stride - padding + k_w;

                    if (d_in >= 0 && d_in < depth_in && h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                        int i = n * in_channels * depth_in * height_in * width_in +
                                c_in * depth_in * height_in * width_in +
                                d_in * height_in * width_in +
                                h_in * width_in +
                                w_in;
                        int j = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                                c_in * kernel_size * kernel_size * kernel_size +
                                k_d * kernel_size * kernel_size +
                                k_h * kernel_size +
                                k_w;
                        sum += input[i] * weight[j];
                    }
                }
            }
        }
    }

    output[n * out_channels * depth_out * height_out * width_out +
           c_out * depth_out * height_out * width_out +
           d_out * height_out * width_out +
           (height_out - 1) * stride - padding +
           (width_out - 1) * stride - padding] = sum;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto depth_out = (depth_in - 1) * stride + kernel_size - 2 * padding;
    auto height_out = (height_in - 1) * stride + kernel_size - 2 * padding;
    auto width_out = (width_in - 1) * stride + kernel_size - 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 16;
    const int num_blocks_x = (width_out + block_size - 1) / block_size;
    const int num_blocks_y = (height_out + block_size - 1) / block_size;
    const int num_blocks_z = (batch_size + block_size - 1) / block_size;

    conv_transpose_3d_kernel<<<num_blocks_x * num_blocks_y * num_blocks_z, block_size * block_size * block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding);

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


# Define the custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* input, float* mean, float* var, float* gamma, float* beta, float* output, int batch_size, int channels, int depth, int height, int width, int group_size) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || g >= group_size || c >= channels) return;

    float sum = 0.0f;
    float sum_sqr = 0.0f;
    int count = depth * height * width;

    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int i = n * channels * depth * height * width +
                        c * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
                sum += input[i];
                sum_sqr += input[i] * input[i];
            }
        }
    }

    mean[g * channels + c] = sum / count;
    var[g * channels + c] = sum_sqr / count - mean[g * channels + c] * mean[g * channels + c];

    __syncthreads();

    if (threadIdx.x == 0) {
        mean[g * channels + c] /= group_size;
        var[g * channels + c] /= group_size;
    }

    __syncthreads();

    float inv_std = rsqrt(var[g * channels + c] + 1e-5);
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int i = n * channels * depth * height * width +
                        c * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
                output[i] = gamma[g * channels + c] * (input[i] - mean[g * channels + c]) * inv_std + beta[g * channels + c];
            }
        }
    }
}

torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int group_size) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto mean = torch::zeros({channels}, input.options());
    auto var = torch::zeros({channels}, input.options());
    auto output = torch::zeros_like(input);

    const int block_size = 16;
    const int num_blocks_x = (width + block_size - 1) / block_size;
    const int num_blocks_y = (height + block_size - 1) / block_size;
    const int num_blocks_z = (batch_size * channels + block_size - 1) / block_size;

    group_norm_kernel<<<num_blocks_x * num_blocks_y * num_blocks_z, block_size * block_size * block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, depth, height, width, group_size);

    return output;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int group_size);"
)

# Compile the inline CUDA code for Group Normalization
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.group_norm = group_norm

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_cuda(x, self.weight)
        x = torch.sigmoid(x) * x  # Swish activation
        x = self.group_norm.group_norm_cuda(x, self.gamma, self.beta, self.groups)
        x = torch.nn.functional.hardswish(x)  # HardSwish activation
        return x