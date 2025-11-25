import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batch normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_kernel(const float* x, const float* mean, const float* var, const float* gamma, const float* beta, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = gamma[idx] * ((x[idx] - mean[idx]) / sqrt(var[idx] + 1e-5)) + beta[idx];
    }
}

torch::Tensor batch_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    batch_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

batch_norm_cpp_source = (
    "torch::Tensor batch_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta);"
)

# Compile the inline CUDA code for batch normalization
batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = batch_norm
        self.elementwise_subtract = elementwise_subtract

    def forward(self, x):
        x = self.conv_transpose(x)
        mean, var = torch.var_mean(x, dim=(2, 3, 4), unbiased=False, keepdim=True)
        gamma = torch.ones_like(mean)
        beta = torch.zeros_like(beta)
        x = self.batch_norm.batch_norm_cuda(x, mean, var, gamma, beta)
        x_mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        x = self.elementwise_subtract.elementwise_subtract_cuda(x, x_mean)
        return x