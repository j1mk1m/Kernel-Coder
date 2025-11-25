import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for applying Mish twice
mish_twice_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline float mish(float x) {
    float exp_x = expf(x);
    float softplus = logf(1.0f + exp_x);
    return x * tanhf(softplus);
}

__global__ void mish_twice_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float temp = mish(x);
        output[idx] = mish(temp);
    }
}

torch::Tensor mish_twice_cuda(torch::Tensor input) {
    int64_t size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    mish_twice_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    
    return output;
}
"""

mish_twice_cpp_source = """
torch::Tensor mish_twice_cuda(torch::Tensor input);
"""

# Compile the CUDA code
mish_twice = load_inline(
    name="mish_twice",
    cpp_sources=mish_twice_cpp_source,
    cuda_sources=mish_twice_source,
    functions=["mish_twice_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.mish_twice = mish_twice  # The loaded CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.mish_twice.mish_twice_cuda(x)
        return x

batch_size = 64  
in_channels = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]