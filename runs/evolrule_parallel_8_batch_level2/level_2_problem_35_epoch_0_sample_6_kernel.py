import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Subtract and Hardswish CUDA kernel
subtract_hardswish_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void subtract_hardswish_kernel(const float* input, float subtract_val, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_val;
        float temp = x + 3.0f;
        temp = fmaxf(0.0f, fminf(temp, 6.0f)); // Apply ReLU6
        output[idx] = x * temp / 6.0f;
    }
}

torch::Tensor subtract_hardswish_cuda(torch::Tensor input, float subtract_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    subtract_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), subtract_val,
        output.data_ptr<float>(), size
    );
    
    return output;
}
"""

subtract_hardswish_cpp = "torch::Tensor subtract_hardswish_cuda(torch::Tensor input, float subtract_val);"
subtract_hardswish = load_inline(
    name="subtract_hardswish",
    cpp_sources=subtract_hardswish_cpp,
    cuda_sources=subtract_hardswish_source,
    functions=["subtract_hardswish_cuda"],
    verbose=True,
)

# Mish Activation CUDA kernel
mish_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float exp_x = expf(x);
        float softplus = logf(1.0f + exp_x);
        float tanh_softplus = tanhf(softplus);
        output[idx] = x * tanh_softplus;
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    
    return output;
}
"""

mish_cpp = "torch::Tensor mish_cuda(torch::Tensor input);"
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
    
    def forward(self, x):
        x = self.conv(x)
        # Fused Subtract + Hardswish
        x = subtract_hardswish.subtract_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        # Custom Mish activation
        x = mish.mish_cuda(x)
        return x