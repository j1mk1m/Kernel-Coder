import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise operations after convolution
elementwise_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elementwise_operations(const float* input, float* output, int num_elements, float add_val, float multiply_val) {
    const float sqrt2 = 1.4142135623730950488016887242097f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float temp = input[idx] + add_val;
        // Clamp to maximum 0.0
        temp = (temp < 0.0f) ? temp : 0.0f;
        // Compute GELU using erf approximation
        float x = temp;
        float x_over_sqrt2 = x / sqrt2;
        float erf_val = erf(x_over_sqrt2);
        float gelu_val = 0.5f * x * (1.0f + erf_val);
        output[idx] = gelu_val * multiply_val;
    }
}

torch::Tensor elementwise_ops_cuda(torch::Tensor input, float add_val, float multiply_val) {
    auto output = torch::empty_like(input);
    int64_t num_elements = input.numel();
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    elementwise_operations<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        add_val,
        multiply_val
    );

    return output;
}
"""

elementwise_ops_header = """
torch::Tensor elementwise_ops_cuda(torch::Tensor input, float add_val, float multiply_value);
"""

# Compile the inline CUDA code
elementwise_ops = load_inline(
    name="elementwise_ops",
    cpp_sources=elementwise_ops_header,
    cuda_sources=elementwise_ops_source,
    functions=["elementwise_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.elementwise_ops = elementwise_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply fused element-wise operations using custom CUDA kernel
        return self.elementwise_ops.elementwise_ops_cuda(
            x, self.add_value, self.multiply_value
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]