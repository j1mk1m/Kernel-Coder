import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_ops_kernel(const float* x, float add_val, float multiply_val, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float tmp = x[idx] + add_val;
        tmp = min(tmp, 0.0f);
        float z = tmp / 1.41421356237f;
        tmp = 0.5f * tmp * (1.0f + erf(z));
        tmp *= multiply_val;
        out[idx] = tmp;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, float add_val, float multiply_val) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), add_val, multiply_val, out.data_ptr<float>(), size);
    return out;
}
"""

fused_ops_cpp = (
    "torch::Tensor fused_ops_cuda(torch::Tensor x, float add_val, float multiply_val);"
)

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_ops.fused_ops_cuda(x, self.add_value, self.multiply_value)

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]