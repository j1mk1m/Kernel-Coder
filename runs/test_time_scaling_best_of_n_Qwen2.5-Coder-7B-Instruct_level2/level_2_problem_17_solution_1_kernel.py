import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int input_height, int input_width, int input_channels, int output_height, int output_width, int output_channels, int kernel_size) {
    int batch_idx = blockIdx.x / (output_height * output_width);
    int output_channel_idx = blockIdx.x % (output_height * output_width);
    int output_height_idx = output_channel_idx / output_width;
    int output_width_idx = output_channel_idx % output_width;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int input_channel_idx = output_channel_idx;
            int input_height_idx = output_height_idx * stride + i - padding;
            int input_width_idx = output_width_idx * stride + j - padding;

            if (input_height_idx >= 0 && input_height_idx < input_height && input_width_idx >= 0 && input_width_idx < input_width) {
                sum += input[batch_idx * input_height * input_width * input_channels + input_height_idx * input_width * input_channels + input_width_idx * input_channels + input_channel_idx] *
                       weight[input_channel_idx * kernel_size * kernel_size + i * kernel_size + j];
            }
        }
    }

    output[batch_idx * output_height * output_width * output_channels + output_height_idx * output_width * output_channels + output_width_idx * output_channels + output_channel_idx] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto input_channels = input.size(1);
    auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    auto output_channels = weight.size(0);

    auto output = torch::zeros({input.size(0), output_channels, output_height, output_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (output_height * output_width * output_channels + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), input_height, input_width, input_channels, output_height, output_width, output_channels, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
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


# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int channels, int height, int width) {
    int channel_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int total_elements = batch_size * height * width;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = thread_idx; i < total_elements; i += blockDim.x) {
        int batch_idx = i / (height * width);
        int height_idx = (i / width) % height;
        int width_idx = i % width;
        sum += input[batch_idx * channels * height * width + channel_idx * height * width + height_idx * width + width_idx];
        sum_sq += input[batch_idx * channels * height * width + channel_idx * height * width + height_idx * width + width_idx] * input[batch_idx * channels * height * width + channel_idx * height * width + height_idx * width + width_idx];
    }
    __syncthreads();

    sum = block_reduce_sum(sum);
    sum_sq = block_reduce_sum(sum_sq);

    if (thread_idx == 0) {
        mean[channel_idx] = sum / total_elements;
        var[channel_idx] = sum_sq / total_elements - mean[channel_idx] * mean[channel_idx];
    }
    __syncthreads();

    float inv_var = 1.0f / sqrt(var[channel_idx] + eps);
    for (int i = thread_idx; i < total_elements; i += blockDim.x) {
        int batch_idx = i / (height * width);
        int height_idx = (i / width) % height;
        int width_idx = i % width;
        output[batch_idx * channels * height * width + channel_idx * height * width + height_idx * width + width_idx] = (input[batch_idx * channels * height * width + channel_idx * height * width + height_idx * width + width_idx] - mean[channel_idx]) * inv_var;
    }
}

__device__ float block_reduce_sum(float val) {
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            val += __shfl_down_sync(0xFFFFFFFF, val, s);
        }
    }
    return val;
}

const float eps = 1e-5f;

torch::Tensor instance_norm_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto mean = torch::zeros({channels}, input.options());
    auto var = torch::zeros({channels}, input.options());
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * height * width + block_size - 1) / block_size;

    instance_norm_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.instance_norm = instance_norm

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, stride=1, padding=1)
        x = self.instance_norm.instance_norm_cuda(x)
        x = x / divide_by
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.weight = state_dict[prefix + 'conv.weight']

# Example usage:
model_new = ModelNew(in_channels, out_channels, kernel_size, divide_by)
inputs = get_inputs()[0].cuda()
outputs = model_new(inputs)
print(outputs.shape)