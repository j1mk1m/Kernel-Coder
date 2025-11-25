import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Mish-Mish CUDA kernel
mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_mish_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        // First Mish activation
        float soft1;
        if (x > 20.0f) {
            soft1 = x;
        } else if (x < -20.0f) {
            soft1 = expf(x);
        } else {
            if (x > 0) {
                soft1 = x + log1pf(expf(-x));
            } else {
                soft1 = log1pf(expf(x));
            }
        }
        float mish1 = x * tanhf(soft1);

        // Second Mish activation
        float soft2;
        if (mish1 > 20.0f) {
            soft2 = mish1;
        } else if (mish1 < -20.0f) {
            soft2 = expf(mish1);
        } else {
            if (mish1 > 0) {
                soft2 = mish1 + log1pf(expf(-mish1));
            } else {
                soft2 = log1pf(expf(mish1));
            }
        }
        out[idx] = mish1 * tanhf(soft2);
    }
}

torch::Tensor mish_mish_cuda(torch::Tensor in) {
    auto size = in.numel();
    auto out = torch::zeros_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_mish_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

mish_mish_cpp_header = """
torch::Tensor mish_mish_cuda(torch::Tensor in);
"""

# Compile the fused Mish-Mish CUDA kernel
mish_mish_cpp = load_inline(
    name="mish_mish",
    cpp_sources=mish_mish_cpp_header,
    cuda_sources=mish_mish_source,
    functions=["mish_mish_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = mish_mish_cpp.mish_mish_cuda(x)
        return x

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]