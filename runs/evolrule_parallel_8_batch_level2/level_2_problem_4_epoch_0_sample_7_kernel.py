import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

# Define the fused Mish CUDA kernel
double_mish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void double_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float s1 = logf(1.0f + expf(x));
        float y = x * tanhf(s1);
        float s2 = logf(1.0f + expf(y));
        output[idx] = y * tanhf(s2);
    }
}

torch::Tensor double_mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    double_mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

double_mish_cpp_source = "torch::Tensor double_mish_cuda(torch::Tensor input);"

# Compile the fused Mish kernel
double_mish = load_inline(
    name="double_mish",
    cpp_sources=double_mish_cpp_source,
    cuda_sources=double_mish_source,
    functions=["double_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.double_mish = double_mish  # The fused Mish kernel

    def forward(self, x):
        x = self.conv(x)
        x = self.double_mish.double_mish_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]