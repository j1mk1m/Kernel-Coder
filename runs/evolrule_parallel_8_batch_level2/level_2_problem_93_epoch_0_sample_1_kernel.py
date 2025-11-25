import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_ops_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <math.h>

#define PI 3.14159265358979323846f

__device__ __forceinline__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float poly = 0.044715f;
    float inner = x + poly * x * x * x;
    float tanh_val = tanhf(sqrt_2_over_pi * inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void fused_operations_kernel(const float* input, float* output, 
                                       int size, float add_val, float mult_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] + add_val;
        val = fminf(val, 0.0f);  // clamp to max 0.0
        val = compute_gelu(val);
        output[idx] = val * mult_val;
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, float add_val, float mult_val) {
    const int64_t size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    // Launch the kernel
    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size, 
        add_val, 
        mult_val
    );
    
    return output;
}
"""

# The header for the CPP code (required for load_inline)
fused_ops_cpp_source = (
    "torch::Tensor fused_operations_cuda(torch::Tensor input, float add_val, float mult_val);"
)

# Compile the CUDA code
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=[fused_ops_cpp_source],
    cuda_sources=[fused_ops_source],
    functions=["fused_operations_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_operations = fused_operations  # loaded kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_operations.fused_operations_cuda(x, self.add_value, self.multiply_value)
        return x