import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int D, int H, int W, int kD, int kH, int kW) {
    int b = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x / W;
    int w_out = blockIdx.x % W;

    int d_in_start = max(h_out * stride_h - pad_h, 0);
    int d_in_end = min(d_in_start + kD, D);
    int h_in_start = max(w_out * stride_w - pad_w, 0);
    int h_in_end = min(h_in_start + kH, H);
    int w_in_start = max(c_out * stride_d - pad_d, 0);
    int w_in_end = min(w_in_start + kW, W);

    float sum = 0.0f;
    for (int d_in = d_in_start; d_in < d_in_end; ++d_in) {
        for (int h_in = h_in_start; h_in < h_in_end; ++h_in) {
            for (int w_in = w_in_start; w_in < w_in_end; ++w_in) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    sum += input[b * in_channels * D * H * W + c_in * D * H * W + d_in * H * W + h_in * W + w_in] *
                           weight[c_out * in_channels * kD * kH * kW + c_in * kD * kH * kW + (d_in - d_in_start) * kH * kW + (h_in - h_in_start) * kW + (w_in - w_in_start)];
                }
            }
        }
    }

    output[b * out_channels * H * W + c_out * H * W + h_out * W + w_out] = sum;
}

torch::Tensor conv_3d_cuda(torch::Tensor input, torch::Tensor weight, int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto kD = weight.size(2);
    auto kH = weight.size(3);
    auto kW = weight.size(4);

    auto output = torch::zeros({batch_size, out_channels, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * H * W + block_size - 1) / block_size;

    conv_3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, D, H, W, kD, kH, kW);

    return output;
}
"""

conv_3d_cpp_source = (
    "torch::Tensor conv_3d_cuda(torch::Tensor input, torch::Tensor weight, int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w);"
)

# Compile the inline CUDA code for 3D convolution
conv_3d = load_inline(
    name="conv_3d",
    cpp_sources=conv_3d_cpp_source,
    cuda_sources=conv_3d_source,
    functions=["conv_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for minimum operation
min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_kernel(const float* input, float* output, int batch_size, int in_channels, int D, int H, int W, int dim) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int h_out = blockIdx.x / W;
    int w_out = blockIdx.x % W;

    int idx_start = dim == 0 ? 0 : h_out * W + w_out;
    int idx_end = dim == 0 ? H * W : dim == 1 ? W : 1;

    float min_val = INFINITY;
    for (int i = idx_start; i < idx_end; ++i) {
        min_val = fmin(min_val, input[b * in_channels * D * H * W + c * D * H * W + i]);
    }

    output[b * in_channels * H * W + c * H * W + h_out * W + w_out] = min_val;
}

torch::Tensor min_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::zeros({batch_size, in_channels, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * in_channels * H * W + block_size - 1) / block_size;

    min_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, D, H, W, dim);

    return output;
}
"""

min_cpp_source = (
    "torch::Tensor min_cuda(torch::Tensor input, int dim);"
)

# Compile the inline CUDA code for minimum operation
min_op = load_inline(
    name="min_op",
    cpp_sources=min_cpp_source,
    cuda_sources=min_source,
    functions=["min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int in_channels, int H, int W) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int h = blockIdx.x / W;
    int w = blockIdx.x % W;

    int idx = c * H * W + h * W + w;
    float exp_val = exp(input[b * in_channels * H * W + idx]);
    float sum_exp = 0.0f;

    for (int i = 0; i < in_channels; ++i) {
        sum_exp += exp(input[b * in_channels * H * W + i * H * W + h * W + w]);
    }

    output[b * in_channels * H * W + idx] = exp_val / sum_exp;
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::zeros({batch_size, in_channels, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * in_channels * H * W + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, H, W);

    return output;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = conv_3d
        self.min_op = min_op
        self.softmax = softmax

    def forward(self, x):
        x = self.conv.conv_3d_cuda(x, self.weight, stride_d=1, stride_h=1, stride_w=1, pad_d=1, pad_h=1, pad_w=1)
        x = self.min_op.min_cuda(x, dim=self.dim)
        x = self.softmax.softmax_cuda(x)
        return x