import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for layer normalization
layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layer_norm_forward_kernel(const float* x, const float* mean, const float* inv_std, float* y, int batch_size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * spatial_size) {
        int c_idx = idx / (spatial_size);
        int s_idx = idx % (spatial_size);
        y[idx] = (x[idx] - mean[c_idx]) * inv_std[c_idx];
    }
}

torch::Tensor layer_norm_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor inv_std) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto spatial_size = x.numel() / (batch_size * channels);

    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;

    layer_norm_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), inv_std.data_ptr<float>(), y.data_ptr<float>(), batch_size, channels, spatial_size);

    return y;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_forward_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor inv_std);"
)

# Compile the inline CUDA code for layer normalization
layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_forward_kernel(const float* x, float* y, int batch_size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * spatial_size) {
        int c_idx = idx / (spatial_size);
        int s_idx = idx % (spatial_size);
        float tanh_val = tanh(0.7978845608 * (x[idx] + 0.044715 * x[idx] * x[idx] * x[idx]));
        y[idx] = 0.5 * x[idx] * (1.0 + tanh_val);
    }
}

torch::Tensor gelu_forward_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto spatial_size = x.numel() / (batch_size * channels);

    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;

    gelu_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, channels, spatial_size);

    return y;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_forward_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for GELU activation
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = layer_norm
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        mean = torch.mean(x, dim=(1, 2, 3))
        inv_std = 1.0 / torch.sqrt(torch.var(x, dim=(1, 2, 3)) + eps)
        x = self.layer_norm.layer_norm_forward_cuda(x, mean, inv_std)
        x = gelu.gelu_backward_cuda(x, x)
        x = x * self.scaling_factor
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]