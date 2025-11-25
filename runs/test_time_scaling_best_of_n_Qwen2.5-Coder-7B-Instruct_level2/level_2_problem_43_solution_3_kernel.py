import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

convolution_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < batch_size && c_out < out_channels && d < depth) {
        float sum = 0.0f;
        for (int i = 0; i < in_channels; ++i) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int j = 0; j < kernel_size; ++j) {
                    for (int l = 0; l < kernel_size; ++l) {
                        int d_in = d + k - kernel_size / 2;
                        int h_in = i + j - kernel_size / 2;
                        int w_in = i + l - kernel_size / 2;
                        if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                            sum += input[(n * in_channels + i) * depth * height * width + d_in * height * width + h_in * width + w_in] * weight[(c_out * in_channels + i) * kernel_size * kernel_size * kernel_size + k * kernel_size * kernel_size + j * kernel_size + l];
                        }
                    }
                }
            }
        }
        output[(n * out_channels + c_out) * depth + d] = sum;
    }
}

torch::Tensor convolution_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, depth}, torch::device("cuda"));

    const int block_size = 32;
    const int num_blocks_d = (depth + block_size - 1) / block_size;
    const int num_blocks_n = (batch_size + block_size - 1) / block_size;

    convolution_3d_kernel<<<num_blocks_n, num_blocks_d, 0, at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);

    return output;
}
"""

convolution_3d_cpp_source = (
    "torch::Tensor convolution_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

max_pooling_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pooling_3d_kernel(const float* input, float* output, int batch_size, int in_channels, int depth, int height, int width, int pool_size) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < batch_size && c < in_channels && d < depth) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < pool_size; ++k) {
            for (int j = 0; j < pool_size; ++j) {
                for (int l = 0; l < pool_size; ++l) {
                    int d_in = d + k;
                    int h_in = d + j;
                    int w_in = d + l;
                    if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        max_val = std::max(max_val, input[(n * in_channels + c) * depth * height * width + d_in * height * width + h_in * width + w_in]);
                    }
                }
            }
        }
        output[(n * in_channels + c) * depth + d] = max_val;
    }
}

torch::Tensor max_pooling_3d_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto pool_size = 2;

    auto output = torch::zeros({batch_size, in_channels, depth / pool_size, height / pool_size, width / pool_size}, torch::device("cuda"));

    const int block_size = 32;
    const int num_blocks_d = (depth / pool_size + block_size - 1) / block_size;
    const int num_blocks_n = (batch_size + block_size - 1) / block_size;

    max_pooling_3d_kernel<<<num_blocks_n, num_blocks_d, 0, at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, depth, height, width, pool_size);

    return output;
}
"""

max_pooling_3d_cpp_source = (
    "torch::Tensor max_pooling_3d_cuda(torch::Tensor input);"
)

log_sum_exp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void log_sum_exp_kernel(const float* input, float* output, int batch_size, int channels, int depth) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < batch_size && c < channels && d < depth) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < depth; ++i) {
            max_val = std::max(max_val, input[(n * channels + c) * depth + i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < depth; ++i) {
            sum_exp += exp(input[(n * channels + c) * depth + i] - max_val);
        }

        output[(n * channels + c) * depth + d] = max_val + log(sum_exp);
    }
}

torch::Tensor log_sum_exp_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);

    auto output = torch::zeros({batch_size, channels, depth}, torch::device("cuda"));

    const int block_size = 32;
    const int num_blocks_d = (depth + block_size - 1) / block_size;
    const int num_blocks_n = (batch_size + block_size - 1) / block_size;

    log_sum_exp_kernel<<<num_blocks_n, num_blocks_d, 0, at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, depth);

    return output;
}
"""

log_sum_exp_cpp_source = (
    "torch::Tensor log_sum_exp_cuda(torch::Tensor input);"
)

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int batch_size, int channels, int depth) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < batch_size && c < channels && d < depth) {
        output[(n * channels + c) * depth + d] = fmaxf(input[(n * channels + c) * depth + d], 0.0f);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);

    auto output = torch::zeros({batch_size, channels, depth}, torch::device("cuda"));

    const int block_size = 32;
    const int num_blocks_d = (depth + block_size - 1) / block_size;
    const int num_blocks_n = (batch_size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks_n, num_blocks_d, 0, at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, depth);

    return output;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for convolution, max pooling, log sum exp, and ReLU
convolution_3d = load_inline(
    name="convolution_3d",
    cpp_sources=convolution_3d_cpp_source,
    cuda_sources=convolution_3d_source,
    functions=["convolution_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

max_pooling_3d = load_inline(
    name="max_pooling_3d",
    cpp_sources=max_pooling_3d_cpp_source,
    cuda_sources=max_pooling_3d_source,
    functions=["max_pooling_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

log_sum_exp = load_inline(
    name="log_sum_exp",
    cpp_sources=log_sum_exp_cpp_source,
    cuda_sources=log_sum_exp_source,
    functions=["log_sum_exp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = convolution_3d
        self.max_pool = max_pooling_3d

    def forward(self, x):
        x = self.conv.convolution_3d_cuda(x, self.weight)
        x = self.max_pool.max_pooling_3d_cuda(x)
        x = log_sum_exp.log_sum_exp_cuda(x)
        x = relu.relu_cuda(x)
        return x