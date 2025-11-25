import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Transposed Convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out, int kernel_size, int stride, int padding) {
    int n = blockIdx.x / (D_out * H_out * W_out);
    int d = blockIdx.x % (D_out * H_out * W_out) / (H_out * W_out);
    int h = blockIdx.x % (H_out * W_out) / W_out;
    int w = blockIdx.x % W_out;

    int in_d_start = max(d * stride - padding, 0);
    int in_d_end = min(in_d_start + kernel_size, D_in);
    int in_h_start = max(h * stride - padding, 0);
    int in_h_end = min(in_h_start + kernel_size, H_in);
    int in_w_start = max(w * stride - padding, 0);
    int in_w_end = min(in_w_start + kernel_size, W_in);

    int out_idx = n * out_channels * D_out * H_out * W_out + d * H_out * W_out + h * W_out + w;

    float sum = 0.0f;
    for (int i = in_d_start; i < in_d_end; ++i) {
        for (int j = in_h_start; j < in_h_end; ++j) {
            for (int k = in_w_start; k < in_w_end; ++k) {
                int in_idx = n * in_channels * D_in * H_in * W_in + ((i - d * stride + padding) / stride) * H_in * W_in + ((j - h * stride + padding) / stride) * W_in + (k - w * stride + padding) / stride;
                int weight_idx = (i - d * stride + padding) * kernel_size * kernel_size * kernel_size + (j - h * stride + padding) * kernel_size * kernel_size + (k - w * stride + padding);
                sum += input[in_idx] * weight[weight_idx];
            }
        }
    }

    output[out_idx] = sum;
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, int D_out, int H_out, int W_out, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * D_out * H_out * W_out + block_size - 1) / block_size;

    conv_transpose_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, D_in, H_in, W_in, D_out, H_out, W_out, kernel_size, stride, padding);

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, int D_out, int H_out, int W_out, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for 3D Transposed Convolution
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int channels, int D, int H, int W) {
    int n = blockIdx.x / (channels * D * H * W);
    int c = blockIdx.x % (channels * D * H * W) / (D * H * W);
    int d = blockIdx.x % (D * H * W) / (H * W);
    int h = blockIdx.x % (H * W) / W;
    int w = blockIdx.x % W;

    int in_idx = n * channels * D * H * W + c * D * H * W + d * H * W + h * W + w;
    float max_val = -INFINITY;
    for (int i = 0; i < channels; ++i) {
        int idx = n * channels * D * H * W + i * D * H * W + d * H * W + h * W + w;
        if (input[idx] > max_val) {
            max_val = input[idx];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < channels; ++i) {
        int idx = n * channels * D * H * W + i * D * H * W + d * H * W + h * W + w;
        sum_exp += exp(input[idx] - max_val);
    }

    output[in_idx] = exp(input[in_idx] - max_val) / sum_exp;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * D * H * W + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, D, H, W);

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

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

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
        self.conv_transpose = conv_transpose
        self.softmax = softmax
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_cuda(x, self.weight, D_out, H_out, W_out, kernel_size, stride, padding)
        x = self.softmax.softmax_cuda(x)
        x = self.sigmoid.sigmoid_cuda(x)
        return x

# Example usage
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

# Initialize the model
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding)
x = get_inputs()[0].cuda()
output = model_new(x)
print(output.shape)