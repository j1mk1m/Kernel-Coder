import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

subtract_and_hardswish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void subtract_and_hardswish_kernel(
    const float* input,
    float subtract_value,
    float* output,
    int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float temp = input[idx] - subtract_value;
        float temp_plus_3 = temp + 3.0f;
        float clamped = fmaxf(0.0f, fminf(temp_plus_3, 6.0f));
        output[idx] = temp * clamped / 6.0f;
    }
}

torch::Tensor subtract_and_hardswish_cuda(
    torch::Tensor input,
    float subtract_value) {
    int64_t num_elements = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    subtract_and_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        subtract_value,
        output.data_ptr<float>(),
        num_elements);
    
    return output;
}
"""

subtract_and_hardswish_cpp_header = "torch::Tensor subtract_and_hardswish_cuda(torch::Tensor input, float subtract_value);"

subtract_and_hardswish = load_inline(
    name="subtract_and_hardswish",
    cpp_sources=subtract_and_hardswish_cpp_header,
    cuda_sources=subtract_and_hardswish_source,
    functions=["subtract_and_hardswish_cuda"],
    verbose=True
)

mish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void mish_kernel(const float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = input[idx];
        float exp_x = expf(x);
        float softplus = logf(1.0f + exp_x);
        float tanh_sp = tanhf(softplus);
        output[idx] = x * tanh_sp;
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    int64_t num_elements = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    
    mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements);
    
    return output;
}
"""

mish_cpp_header = "torch::Tensor mish_cuda(torch::Tensor input);"

mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_header,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.subtract_and_hardswish = subtract_and_hardswish
        self.mish = mish

    def forward(self, x):
        x = self.conv(x)
        x = self.subtract_and_hardswish.subtract_and_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        x = self.mish.mish_cuda(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]