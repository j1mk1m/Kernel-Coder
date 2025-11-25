import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for convolution, average pooling, and sigmoid
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int n, int c_in, int h_in, int w_in, int c_out, int k_h, int k_w) {
    int n_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (n_idx >= n || c_out_idx >= c_out || w_idx >= w_in) {
        return;
    }

    float sum = 0.0f;
    for (int c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                int h_in_idx = w_idx * k_h + kh;
                int w_in_idx = w_idx * k_w + kw;
                int input_idx = n_idx * c_in * h_in * w_in + c_in_idx * h_in * w_in + h_in_idx * w_in + w_in_idx;
                int weight_idx = c_out_idx * c_in * k_h * k_w + c_in_idx * k_h * k_w + kh * k_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    int output_idx = n_idx * c_out * w_in + c_out_idx * w_in + w_idx;
    output[output_idx] = sum;
}
"""

average_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void average_pooling_kernel(const float* input, float* output, int n, int c, int h_in, int w_in, int h_out, int w_out, int stride) {
    int n_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int c_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (n_idx >= n || c_idx >= c || w_idx >= w_out) {
        return;
    }

    float sum = 0.0f;
    int count = 0;
    for (int kh = 0; kh < stride; ++kh) {
        for (int kw = 0; kw < stride; ++kw) {
            int h_in_idx = w_idx * stride + kh;
            int w_in_idx = w_idx * stride + kw;
            int input_idx = n_idx * c * h_in * w_in + c_idx * h_in * w_in + h_in_idx * w_in + w_in_idx;
            sum += input[input_idx];
            count++;
        }
    }
    int output_idx = n_idx * c * w_out + c_idx * w_out + w_idx;
    output[output_idx] = sum / count;
}
"""

sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel(float* input, int n, int c, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n * c * h * w) {
        return;
    }

    input[idx] = 1.0f / (1.0f + exp(-input[idx]));
}
"""

# Compile the inline CUDA code for convolution, average pooling, and sigmoid
convolution_module = load_inline(
    name="convolution",
    cpp_sources="",
    cuda_sources=convolution_source,
    functions=["convolution_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

average_pooling_module = load_inline(
    name="average_pooling",
    cpp_sources="",
    cuda_sources=average_pooling_source,
    functions=["average_pooling_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

sigmoid_module = load_inline(
    name="sigmoid",
    cpp_sources="",
    cuda_sources=sigmoid_source,
    functions=["sigmoid_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Initialize weights for convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # Convolution
        conv_output = torch.zeros((x.size(0), self.out_channels, x.size(2) - self.kernel_size + 1, x.size(3) - self.kernel_size + 1))
        convolution_module.convolution_kernel(
            x.contiguous().view(-1),
            self.weight.view(-1),
            conv_output.contiguous().view(-1),
            x.size(0),
            self.in_channels,
            x.size(2),
            x.size(3),
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
        )

        # Average Pooling
        avg_pool_output = torch.zeros((x.size(0), self.out_channels, x.size(2) // self.pool_kernel_size, x.size(3) // self.pool_kernel_size))
        average_pooling_module.average_pooling_kernel(
            conv_output.contiguous().view(-1),
            avg_pool_output.contiguous().view(-1),
            x.size(0),
            self.out_channels,
            x.size(2),
            x.size(3),
            x.size(2) // self.pool_kernel_size,
            x.size(3) // self.pool_kernel_size,
            self.pool_kernel_size,
        )

        # Sigmoid
        sigmoid_output = torch.zeros_like(avg_pool_output)
        sigmoid_module.sigmoid_kernel(
            avg_pool_output.contiguous().view(-1),
            avg_pool_output.size(0) * avg_pool_output.size(1) * avg_pool_output.size(2) * avg_pool_output.size(3),
        )
        sigmoid_output.copy_(avg_pool_output)

        # Sum over all spatial dimensions
        output = torch.sum(sigmoid_output, dim=[1, 2, 3])

        return output

# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 8
    out_channels = 64
    height, width = 384, 384
    kernel_size = 3
    pool_kernel_size = 4

    model_new = ModelNew(in_channels, out_channels, kernel_size, pool_kernel_size)
    inputs = torch.rand(batch_size, in_channels, height, width).cuda()
    outputs = model_new(inputs)
    print(outputs.shape)