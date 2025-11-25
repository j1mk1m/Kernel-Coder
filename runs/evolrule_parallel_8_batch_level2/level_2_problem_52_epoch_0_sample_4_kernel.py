import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused activation kernel: tanh(softplus(x)) * x
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activation_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float softplus = log(1 + exp(x[idx]));
        out[idx] = tanh(softplus) * x[idx];
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor x);"
)

# Compile the fused activation kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(x)
        x = self.bn(x)
        return x

# Keep these functions as in the original
batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]