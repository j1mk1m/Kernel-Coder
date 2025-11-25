import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

# Define the fused activation CUDA kernel
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activation_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x_val = x[idx];
    float s;
    if (x_val > 20.0f) {
        s = x_val;
    } else if (x_val < -20.0f) {
        s = 0.0f;
    } else {
        float exp_x = expf(x_val);
        s = log1pf(exp_x);
    }
    float t = tanhf(s);
    out[idx] = x_val * t;
}

torch::Tensor fused_activation_cuda(torch::Tensor x) {
    AT_ASSERTM(x.is_cuda(), "Input must be a CUDA tensor");
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 256;
    int blocks = (size + threads - 1) / threads;

    fused_activation_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_activation_header = "torch::Tensor fused_activation_cuda(torch::Tensor x);"

# Load the fused activation kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_header,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.fused_activation = fused_activation  # Access the loaded module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(x)
        x = self.bn(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]