import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and division CUDA kernel
fused_relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_div_kernel(
    const float* input, float* output, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = fmaxf(val, 0.0f);  // Apply ReLU
        output[idx] = val / divisor;  // Divide by constant
    }
}

torch::Tensor fused_relu_div_cuda(torch::Tensor input, float divisor) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_relu_div_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        divisor, 
        size);
    return output;
}
"""

fused_relu_div_cpp_source = (
    "torch::Tensor fused_relu_div_cuda(torch::Tensor input, float divisor);"
)

# Compile the fused ReLU/division kernel
fused_relu_div = load_inline(
    name="fused_relu_div",
    cuda_sources=fused_relu_div_source,
    cpp_sources=fused_relu_div_cpp_source,
    functions=["fused_relu_div_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.fused_relu_div = fused_relu_div  # Store the kernel wrapper

    def forward(self, x):
        x = self.linear(x)  # Compute matmul + bias using cuBLAS
        x = self.fused_relu_div.fused_relu_div_cuda(x, self.divisor)
        return x

# Compatibility with original inputs (no changes needed)
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, divisor]