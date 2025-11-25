import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Load the custom CUDA kernels
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" {
    torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);
}
"""

batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" {
    torch::Tensor batch_norm2d_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta);
}
"""

group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" {
    torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int G);
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);"
)

batch_norm_cpp_source = (
    "torch::Tensor batch_norm2d_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta);"
)

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int G);"
)

conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = conv_transpose.conv_transpose2d_cuda(x, self.conv_transpose.weight, kernel_size=self.conv_transpose.kernel_size[0], stride=self.conv_transpose.stride[0], padding=self.conv_transpose.padding[0])
        x = self.batch_norm(x)
        x = batch_norm.batch_norm2d_cuda(x, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight, self.batch_norm.bias)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        x = group_norm.group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups)
        return x