import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_convolution_kernel(const float* input, float* output, int batch_size, int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    int batch_id = blockIdx.y * blockDim.y + threadIdx.y;
    int channel_id = blockIdx.z * blockDim.z + threadIdx.z;

    if (batch_id >= batch_size || channel_id >= out_channels) {
        return;
    }

    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            int input_i = i * stride - padding;
            int input_j = j * stride - padding;

            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    int input_index = (batch_id * in_channels + channel_id) * input_height * input_width +
                                     (input_i + k) * input_width + (input_j + l);
                    int output_index = (batch_id * out_channels + channel_id) * output_height * output_width +
                                      i * output_width + j;
                    output[output_index] += input[input_index];
                }
            }
        }
    }
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto output = torch::zeros({batch_size, out_channels, input_height, input_width}, input.options());

    const int block_size = 32;
    const int num_threads_x = std::min(output_width, block_size);
    const int num_threads_y = std::min(output_height, block_size);
    const int num_threads_z = std::min(out_channels, block_size);
    const int num_blocks_x = (output_width + num_threads_x - 1) / num_threads_x;
    const int num_blocks_y = (output_height + num_threads_y - 1) / num_threads_y;
    const int num_blocks_z = (out_channels + num_threads_z - 1) / num_threads_z;

    transposed_convolution_kernel<<<num_blocks_x * num_blocks_y * num_blocks_z, num_threads_x * num_threads_y * num_threads_z>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, kernel_size, stride, padding);

    return output;
}
"""

# Define the custom CUDA kernel for log-sum-exp
log_sum_exp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void log_sum_exp_kernel(const float* input, float* output, int batch_size, int channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels) {
        return;
    }

    int batch_id = index / channels;
    int channel_id = index % channels;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < channels; ++i) {
        float val = input[(batch_id * channels + i) * input_height * input_width + (channel_id * input_height * input_width)];
        if (val > max_val) {
            max_val = val;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < channels; ++i) {
        sum += exp(input[(batch_id * channels + i) * input_height * input_width + (channel_id * input_height * input_width)] - max_val);
    }

    output[index] = max_val + log(sum);
}

torch::Tensor log_sum_exp_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output = torch::zeros({batch_size, channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    log_sum_exp_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels);

    return output;
}
"""

# Define the custom CUDA kernel for sum
sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* output, int batch_size, int channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels) {
        return;
    }

    int batch_id = index / channels;
    int channel_id = index % channels;

    float sum = 0.0f;
    for (int i = 0; i < input_height * input_width; ++i) {
        sum += input[(batch_id * channels + channel_id) * input_height * input_width + i];
    }

    output[index] = sum;
}

torch::Tensor sum_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output = torch::zeros({batch_size, channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    sum_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels);

    return output;
}
"""

# Define the custom CUDA kernel for multiplication
multiplication_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void multiplication_kernel(const float* input, float* output, int batch_size, int channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * channels) {
        return;
    }

    int batch_id = index / channels;
    int channel_id = index % channels;

    output[index] = input[index] * 10.0f;
}

torch::Tensor multiplication_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output = torch::zeros({batch_size, channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    multiplication_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels);

    return output;
}
"""

# Compile the inline CUDA code for transposed convolution, log-sum-exp, sum, and multiplication
transposed_convolution = load_inline(
    name="transposed_convolution",
    cpp_sources="",
    cuda_sources=transposed_convolution_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

log_sum_exp = load_inline(
    name="log_sum_exp",
    cpp_sources="",
    cuda_sources=log_sum_exp_source,
    functions=["log_sum_exp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

sum_op = load_inline(
    name="sum_op",
    cpp_sources="",
    cuda_sources=sum_source,
    functions=["sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

multiplication = load_inline(
    name="multiplication",
    cpp_sources="",
    cuda_sources=multiplication_source,
    functions=["multiplication_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.log_sum_exp = log_sum_exp
        self.sum_op = sum_op
        self.multiplication = multiplication

    def forward(self, x):
        x = self.transposed_convolution.transposed_convolution_cuda(x, in_channels, out_channels, kernel_size, 1, 1)  # Transposed convolution
        x = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        x = x + self.bias
        x = self.log_sum_exp.log_sum_exp_cuda(x)  # Log-sum-exp
        x = self.sum_op.sum_cuda(x)  # Sum
        x = self.multiplication.multiplication_cuda(x)  # Multiplication
        return x

# Example usage
if __name__ == "__main__":
    model_new = ModelNew(in_channels, out_channels, kernel_size, bias_shape)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)