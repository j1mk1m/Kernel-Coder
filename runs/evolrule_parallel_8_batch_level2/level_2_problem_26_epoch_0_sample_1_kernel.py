import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused add + hardswish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_add_hswish_kernel(
    const float* a,
    const float* b,
    float* out,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = a[idx] + b[idx];
        float temp = z + 3.0f;
        temp = fmaxf(0.0f, temp);
        temp = fminf(6.0f, temp);
        float h = z * temp / 6.0f;
        out[idx] = z * h;
    }
}

torch::Tensor fused_add_hswish_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hswish_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_add_hswish_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile the fused kernel
fused_mod = load_inline(
    name="fused_add_hswish",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_add_hswish_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_mod  # Store the fused kernel module

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_add_hswish_cuda(x, add_input)
        return x