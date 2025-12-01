import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.x; // batch index
    int c_out = blockIdx.y; // output channel index
    int h_out = blockIdx.z; // output height index
    int w_out = threadIdx.x; // output width index

    int h_in_start = h_out * stride - pad;
    int w_in_start = w_out * stride - pad;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_in_start + kh;
                int w_in = w_in_start + kw;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = n * in_channels * height * width + c_in * height * width + h_in * width + w_in;
                    int weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[n * out_channels * height * width + c_out * height * width + h_out * width + w_out] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int pad) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, (height + 2 * pad - kernel_size) / stride + 1, (width + 2 * pad - kernel_size) / stride + 1}, input.options());

    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

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


# Define the custom CUDA kernel for division
division_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void division_kernel(const float* input, float* output, int numel, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / divisor;
    }
}

torch::Tensor division_cuda(torch::Tensor input, float divisor) {
    auto numel = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    division_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), numel, divisor);

    return output;
}
"""

division_cpp_source = (
    "torch::Tensor division_cuda(torch::Tensor input, float divisor);"
)

# Compile the inline CUDA code for division
division = load_inline(
    name="division",
    cpp_sources=division_cpp_source,
    cuda_sources=division_source,
    functions=["division_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int numel, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] > 0 ? input[idx] : input[idx] * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
    auto numel = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), numel, negative_slope);

    return output;
}
"""

leaky_relu_cpp_source = (
    "torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope);"
)

# Compile the inline CUDA code for LeakyReLU
leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.convolution = convolution
        self.division = division
        self.leaky_relu = leaky_relu

    def forward(self, x):
        x = self.convolution.convolution_cuda(x, self.weight, stride=1, pad=1)
        x = self.division.division_cuda(x, self.divisor)
        x = self.leaky_relu.leaky_relu_cuda(x, negative_slope=0.01)
        return x

    def init_weights(self, in_channels, out_channels, kernel_size):
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()

model_new = ModelNew(in_channels, out_channels, kernel_size, divisor)
model_new.init_weights(in_channels, out_channels, kernel_size)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]