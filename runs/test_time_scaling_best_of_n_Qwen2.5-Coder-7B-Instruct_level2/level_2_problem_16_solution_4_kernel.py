import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] * tanh(softplus(x[idx]));
    }
}

float softplus(float x) {
    return log(1.0f + exp(x));
}

void mish_cuda(float* x, int size) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(x, size);
}
"""

mish_cpp_source = (
    "void mish_cuda(float* x, int size);"
)

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class MishActivation(nn.Module):
    def __init__(self):
        super(MishActivation, self).__init__()

    def forward(self, x):
        mish_cuda(x.view(-1).data_ptr(), x.numel())
        return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.mish = MishActivation()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.mish(x) # Custom Mish activation
        x = x + self.add_value
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1) # Hardtanh activation
        x = x * self.scale # Scaling
        return x