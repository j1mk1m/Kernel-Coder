import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for conv2d
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, float* output, int N, int C_in, int H_in, int W_in, int C_out, int K) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int h_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || c_out >= C_out || h_out >= H_in) return;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_h = 0; k_h < K; ++k_h) {
            for (int k_w = 0; k_w < K; ++k_w) {
                int i = h_out + k_h;
                int j = w_out + k_w;
                if (i >= H_in || j >= W_in) continue;
                sum += input[n * C_in * H_in * W_in + c_in * H_in * W_in + i * W_in + j] * weight[c_out * C_in * K * K + c_in * K * K + k_h * K + k_w];
            }
        }
    }
    output[n * C_out * H_in * W_in + c_out * H_in * W_in + h_out * W_in + w_out] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    auto C_out = weight.size(0);
    auto K = weight.size(2);

    auto output = torch::zeros({N, C_out, H_in, W_in}, input.options());

    const int block_size = 16;
    const int num_blocks = (H_in + block_size - 1) / block_size;

    dim3 grid(num_blocks, num_blocks, N);
    dim3 block(block_size, block_size, 1);

    conv2d_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C_in, H_in, W_in, C_out, K);

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for conv2d
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for batch normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_kernel(const float* input, const float* mean, const float* var, const float* gamma, const float* beta, float* output, int N, int C, int H, int W, float eps) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || c >= C || h >= H) return;

    float normed_input = (input[n * C * H * W + c * H * W + h] - mean[c]) / sqrt(var[c] + eps);
    output[n * C * H * W + c * H * W + h] = gamma[c] * normed_input + beta[c];
}

torch::Tensor bn_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::zeros_like(input);

    const int block_size = 16;
    const int num_blocks = (H + block_size - 1) / block_size;

    dim3 grid(num_blocks, num_blocks, N);
    dim3 block(block_size, block_size, 1);

    bn_kernel<<<grid, block>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, eps);

    return output;
}
"""

bn_cpp_source = (
    "torch::Tensor bn_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps);"
)

# Compile the inline CUDA code for batch normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = conv2d
        self.bn = bn
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

    def forward(self, x):
        x = self.conv.conv2d_cuda(x, self.weight)
        x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
        running_mean = self.running_mean
        running_var = self.running_var
        mean = x.mean((0, 2, 3))
        var = x.var((0, 2, 3), unbiased=False)
        running_mean.mul_(self.momentum).add_(mean * (1 - self.momentum))
        running_var.mul_(self.momentum).add_(var * (1 - self.momentum))
        x = self.bn.bn_cuda(x, running_mean, running_var, self.gamma, self.beta, self.eps)
        return x

    def initialize_parameters(self, in_channels, out_channels, kernel_size):
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))
        self.gamma = torch.nn.Parameter(torch.ones(out_channels))
        self.beta = torch.nn.Parameter(torch.zeros(out_channels))