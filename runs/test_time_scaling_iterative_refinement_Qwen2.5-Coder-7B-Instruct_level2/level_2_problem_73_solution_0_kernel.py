import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x / (width - kernel_size + 1);
    int w_out = blockIdx.x % (width - kernel_size + 1);
    int idx = n * out_channels * height * width + c_out * height * width + h_out * width + w_out;
    output[idx] = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int i_h = h_out + kh;
                int i_w = w_out + kw;
                if (i_h >= 0 && i_h < height && i_w >= 0 && i_w < width) {
                    int i_idx = n * in_channels * height * width + c_in * height * width + i_h * width + i_w;
                    output[idx] += input[i_idx] * weight[c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    auto output = torch::zeros({batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1}, torch::kFloat32);

    dim3 blocks((height - kernel_size + 1) * (width - kernel_size + 1), out_channels, batch_size);
    dim3 threads(1);

    convolution_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size);"
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


# Define the custom CUDA kernel for Batch Normalization
batch_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_normalization_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * height * width) {
        int n = idx / (channels * height * width);
        int c = (idx / (height * width)) % channels;
        int h = (idx / width) % height;
        int w = idx % width;
        float val = input[idx];
        mean[n * channels + c] += val;
        var[n * channels + c] += val * val;
    }
}

void compute_mean_var(float* mean, float* var, int batch_size, int channels, int height, int width) {
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            mean[n * channels + c] /= (height * width);
            var[n * channels + c] /= (height * width);
            var[n * channels + c] -= mean[n * channels + c] * mean[n * channels + c];
        }
    }
}

__global__ void batch_normalization_output_kernel(const float* input, const float* mean, const float* var, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * height * width) {
        int n = idx / (channels * height * width);
        int c = (idx / (height * width)) % channels;
        int h = (idx / width) % height;
        int w = idx % width;
        float val = input[idx];
        output[idx] = (val - mean[n * channels + c]) / sqrt(var[n * channels + c] + 1e-5);
    }
}

torch::Tensor batch_normalization_cuda(torch::Tensor input, int batch_size, int channels, int height, int width) {
    auto mean = torch::zeros({batch_size, channels}, torch::kFloat32);
    auto var = torch::zeros({batch_size, channels}, torch::kFloat32);
    auto output = torch::zeros_like(input);

    dim3 blocks(batch_size * channels * height * width, 1);
    dim3 threads(1);

    batch_normalization_kernel<<<blocks, threads>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    compute_mean_var(mean.data_ptr<float>(), var.data_ptr<float>(), batch_size, channels, height, width);

    batch_normalization_output_kernel<<<blocks, threads>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

batch_normalization_cpp_source = (
    "torch::Tensor batch_normalization_cuda(torch::Tensor input, int batch_size, int channels, int height, int width);"
)

# Compile the inline CUDA code for Batch Normalization
batch_normalization = load_inline(
    name="batch_normalization",
    cpp_sources=batch_normalization_cpp_source,
    cuda_sources=batch_normalization_source,
    functions=["batch_normalization_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bn = batch_normalization
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, batch_size, in_channels, out_channels, height, width, kernel_size)
        x = self.bn.batch_normalization_cuda(x, batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1)
        x = x * self.scaling_factor
        return x