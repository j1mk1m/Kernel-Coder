import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and division CUDA kernel
relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_div_kernel(const float* input, float* output, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = fmaxf(val, 0.f); // Apply ReLU
        output[idx] = val / divisor; // Divide by constant
    }
}

torch::Tensor fused_relu_div_cuda(torch::Tensor input, float divisor) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    fused_relu_div_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        size
    );
    return output;
}
"""

# Compile the CUDA kernel inline
relu_div = load_inline(
    name="fused_relu_div",
    cpp_sources="torch::Tensor fused_relu_div_cuda(torch::Tensor input, float divisor);",
    cuda_sources=relu_div_source,
    functions=["fused_relu_div_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.fused_relu_div = relu_div  # Store the kernel module

    def forward(self, x):
        x = self.linear(x)  # Matrix multiply + bias (using PyTorch's optimized impl)
        x = self.fused_relu_div.fused_relu_div_cuda(x, self.divisor)  # Fused ReLU/division kernel
        return x

def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 8192
    out_features = 8192
    divisor = 2.0
    return [in_features, out_features, divisor]