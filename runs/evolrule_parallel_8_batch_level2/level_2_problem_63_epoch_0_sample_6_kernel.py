import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and division CUDA kernel
fused_relu_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_divide(const float* input, float* output, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]) / divisor;
    }
}

torch::Tensor fused_relu_divide_cuda(torch::Tensor input, float divisor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_divide<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        divisor, 
        size
    );

    return output;
}
"""

fused_relu_divide_cpp = """
torch::Tensor fused_relu_divide_cuda(torch::Tensor input, float divisor);
"""

# Compile the CUDA code
fused_relu_divide = load_inline(
    name="fused_relu_divide",
    cpp_sources=fused_relu_divide_cpp,
    cuda_sources=fused_relu_divide_source,
    functions=["fused_relu_divide_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.fused_relu_divide = fused_relu_divide  # The loaded module

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_relu_divide.fused_relu_divide_cuda(x, self.divisor)
        return x

# Global variables as in the original code
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    # Generate input tensor on CUDA
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    # The arguments needed for __init__
    return [in_features, out_features, divisor]