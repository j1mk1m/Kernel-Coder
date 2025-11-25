import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output, int N, int Cin, int Cout, int D, int H, int W, int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c_out = blockIdx.y * blockDim.x + threadIdx.x;

    if (n >= N || c_out >= Cout) return;

    int d_out_start = max(0, -(pD - kD + sD));
    int h_out_start = max(0, -(pH - kH + sH));
    int w_out_start = max(0, -(pW - kW + sW));

    for (int d_out = 0; d_out < D; ++d_out) {
        int d_in_start = d_out * sD + d_out_start;
        for (int h_out = 0; h_out < H; ++h_out) {
            int h_in_start = h_out * sH + h_out_start;
            for (int w_out = 0; w_out < W; ++w_out) {
                int w_in_start = w_out * sW + w_in_start;
                for (int d_in = max(d_in_start, 0); d_in <= min(D + d_in_start - 1, Cin - 1); ++d_in) {
                    for (int h_in = max(h_in_start, 0); h_in <= min(H + h_in_start - 1, Cin - 1); ++h_in) {
                        for (int w_in = max(w_in_start, 0); w_in <= min(W + w_in_start - 1, Cin - 1); ++w_in) {
                            int in_idx = ((n * Cin + d_in) * H + h_in) * W + w_in;
                            int out_idx = ((n * Cout + c_out) * D + d_out) * H + h_out * W + w_out;
                            atomicAdd(&output[out_idx], input[in_idx] * weight[(c_out * Cin + d_in) * kD * kH * kW + (d_in - d_in_start) * kH * kW + (h_in - h_in_start) * kW + (w_in - w_in_start)]);
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int out_channels, int kernel_size, int stride, int padding) {
    auto N = input.size(0);
    auto Cin = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto Cout = out_channels;
    auto kD = kernel_size;
    auto kH = kernel_size;
    auto kW = kernel_size;
    auto sD = stride;
    auto sH = stride;
    auto sW = stride;
    auto pD = padding;
    auto pH = padding;
    auto pW = padding;

    auto output = torch::zeros({N, Cout, D, H, W}, input.options());

    const int block_size = 16;
    const int num_blocks_x = (Cout + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;

    conv_transpose_3d_kernel<<<num_blocks_y, num_blocks_x>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, Cin, Cout, D, H, W, kD, kH, kW, sD, sH, sW, pD, pH, pW);

    return output;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int out_channels, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for transposed 3D convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for mean pooling
mean_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_pooling_kernel(const float* input, float* output, int N, int Cin, int D, int H, int W, int kD, int kH, int kW, int sD, int sH, int sW) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c = blockIdx.y * blockDim.x + threadIdx.x;

    if (n >= N || c >= Cin) return;

    int d_out_start = max(0, -(kD - sD) / 2);
    int h_out_start = max(0, -(kH - sH) / 2);
    int w_out_start = max(0, -(kW - sW) / 2);

    for (int d_out = 0; d_out < D; ++d_out) {
        int d_in_start = d_out * sD + d_out_start;
        for (int h_out = 0; h_out < H; ++h_out) {
            int h_in_start = h_out * sH + h_out_start;
            for (int w_out = 0; w_out < W; ++w_out) {
                int w_in_start = w_out * sW + w_in_start;
                float sum = 0.0f;
                int count = 0;
                for (int d_in = max(d_in_start, 0); d_in <= min(D + d_in_start - 1, Cin - 1); ++d_in) {
                    for (int h_in = max(h_in_start, 0); h_in <= min(H + h_in_start - 1, Cin - 1); ++h_in) {
                        for (int w_in = max(w_in_start, 0); w_in <= min(W + w_in_start - 1, Cin - 1); ++w_in) {
                            int idx = ((n * Cin + d_in) * H + h_in) * W + w_in;
                            sum += input[idx];
                            count++;
                        }
                    }
                }
                int out_idx = ((n * Cin + c) * D + d_out) * H + h_out * W + w_out;
                output[out_idx] = sum / count;
            }
        }
    }
}

torch::Tensor mean_pooling_cuda(torch::Tensor input, int kernel_size, int stride) {
    auto N = input.size(0);
    auto Cin = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto kD = kernel_size;
    auto kH = kernel_size;
    auto kW = kernel_size;
    auto sD = stride;
    auto sH = stride;
    auto sW = stride;

    auto output = torch::zeros({N, Cin, D, H, W}, input.options());

    const int block_size = 16;
    const int num_blocks_x = (Cin + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;

    mean_pooling_kernel<<<num_blocks_y, num_blocks_x>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, Cin, D, H, W, kD, kH, kW, sD, sH, sW);

    return output;
}
"""

mean_pooling_cpp_source = (
    "torch::Tensor mean_pooling_cuda(torch::Tensor input, int kernel_size, int stride);"
)

# Compile the inline CUDA code for mean pooling
mean_pooling = load_inline(
    name="mean_pooling",
    cpp_sources=mean_pooling_cpp_source,
    cuda_sources=mean_pooling_source,
    functions=["mean_pooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N, int Cin, int D, int H, int W) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c = blockIdx.y * blockDim.x + threadIdx.x;

    if (n >= N || c >= Cin) return;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = ((n * Cin + c) * D + d) * H + h * W + w;
                max_val = std::max(max_val, input[idx]);
            }
        }
    }

    float sum_exp = 0.0f;
    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = ((n * Cin + c) * D + d) * H + h * W + w;
                sum_exp += exp(input[idx] - max_val);
            }
        }
    }

    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = ((n * Cin + c) * D + d) * H + h * W + w;
                output[idx] = exp(input[idx] - max_val) / sum_exp;
            }
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto Cin = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::zeros_like(input);

    const int block_size = 16;
    const int num_blocks_x = (Cin + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks_y, num_blocks_x>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, Cin, D, H, W);

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_cuda(x, self.weight, out_channels, kernel_size, stride, padding)
        x = x.mean(dim=2, keepdim=True)
        x = x + self.bias
        x = softmax.softmax_cuda(x)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        return x

# === Test config ===
batch_size = 16
in_channels = 16
out_channels = 64
depth = 32; height = width = 128
kernel_size = 3
stride = 1
padding = 1
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]