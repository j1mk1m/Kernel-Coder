import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for two Mish operations fused into one kernel
mish_twice_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_twice_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];

    // Compute first Mish: x * exp(x) / (exp(x) + 2)
    float exp_x = expf(x);
    float term1 = exp_x / (exp_x + 2.0f);
    float y = x * term1;

    // Compute second Mish: y * exp(y) / (exp(y) + 2)
    float exp_y = expf(y);
    float term2 = exp_y / (exp_y + 2.0f);
    float z = y * term2;

    output[idx] = z;
}

torch::Tensor mish_twice_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_twice_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

mish_twice_cpp_source = """
torch::Tensor mish_twice_cuda(torch::Tensor input);
"""

# Compile the fused Mish kernel
mish_twice = load_inline(
    name="mish_twice",
    cuda_sources=mish_twice_source,
    cpp_sources=mish_twice_cpp_source,
    functions=["mish_twice_cuda"],
    verbose=True,
    extra_cuda_cflags=["--use_fast_math", "-O3"],
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.mish_twice = mish_twice  # Reference to the compiled CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.mish_twice.mish_twice_cuda(x)
        return x

# Input and initialization functions remain unchanged
batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]