import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and division CUDA kernel
fused_relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_div_inplace(float* data, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        data[idx] = fmaxf(0.0f, val) / divisor;
    }
}

torch::Tensor fused_relu_div_cuda(torch::Tensor input, float divisor) {
    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;

    fused_relu_div_inplace<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        divisor,
        input.numel()
    );

    return input;
}
"""

fused_relu_div_cpp_source = (
    "torch::Tensor fused_relu_div_cuda(torch::Tensor input, float divisor);"
)

# Compile the fused kernel
fused_relu_div = load_inline(
    name="fused_relu_div",
    cpp_sources=fused_relu_div_cpp_source,
    cuda_sources=fused_relu_source,
    functions=["fused_relu_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.fused_relu_div = fused_relu_div  # Load the fused kernel

    def forward(self, x):
        x = self.linear(x)  # Matrix multiply + bias
        x = self.fused_relu_div.fused_relu_div_cuda(x, self.divisor)  # Fused ReLU and division
        return x

def get_inputs():
    # Generate input tensors on CUDA
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    # Initialization parameters (no tensors needed)
    return [in_features, out_features, divisor]