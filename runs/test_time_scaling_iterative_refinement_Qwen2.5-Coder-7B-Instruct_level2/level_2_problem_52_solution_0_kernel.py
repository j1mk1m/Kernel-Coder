import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise inverse hyperbolic tangent
elementwise_arctanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_arctanh_kernel(const float* a, float* out, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = atanh(a[idx]);
    }
}

torch::Tensor elementwise_arctanh_cuda(torch::Tensor a) {
    auto numel = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    elementwise_arctanh_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), numel);

    return out;
}
"""

elementwise_arctanh_cpp_source = (
    "torch::Tensor elementwise_arctanh_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for element-wise inverse hyperbolic tangent
elementwise_arctanh = load_inline(
    name="elementwise_arctanh",
    cpp_sources=elementwise_arctanh_cpp_source,
    cuda_sources=elementwise_arctanh_source,
    functions=["elementwise_arctanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = batch_norm

    def forward(self, x):
        x = self.conv(x)
        x = elementwise_arctanh_cuda(elementwise_add_cuda(relu_cuda(tanh_cuda(softplus_cuda(x))), self.weight))
        x = self.bn.batch_norm_cuda(x, self.weight, self.bias, self.running_mean, self.running_var, eps)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]