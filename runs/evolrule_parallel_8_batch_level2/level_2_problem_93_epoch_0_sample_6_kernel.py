import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused element-wise operations
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_elementwise_kernel(
    const float* input,
    float* output,
    float add_val,
    float multiply_val,
    float sqrt_2,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] + add_val;
        temp = fminf(temp, 0.0f);
        float temp_over_sqrt2 = temp / sqrt_2;
        float erf_val = erf(temp_over_sqrt2);
        float gelu_val = 0.5f * temp * (1.0f + erf_val);
        output[idx] = gelu_val * multiply_val;
    }
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor input,
    float add_val,
    float multiply_val,
    float sqrt_2
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_val,
        multiply_val,
        sqrt_2,
        size
    );

    return output;
}
"""

fused_elementwise_header = """
extern "C" {
    torch::Tensor fused_elementwise_cuda(
        torch::Tensor input,
        float add_val,
        float multiply_val,
        float sqrt_2
    );
}
"""

# Compile the inline CUDA code for the fused element-wise operations
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_header,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.sqrt_2 = math.sqrt(2.0)
        self.fused_elementwise = fused_elementwise

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_elementwise.fused_elementwise_cuda(
            x, self.add_value, self.multiply_value, self.sqrt_2
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]