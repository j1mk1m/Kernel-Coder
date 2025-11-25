import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for sigmoid, scaling, and residual add
fused_sigmoid_scale_add_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_sigmoid_scale_add_kernel(
    const float* input,
    float scaling_factor,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        float scaled = sigmoid_val * scaling_factor;
        output[idx] = scaled + val;
    }
}

torch::Tensor fused_sigmoid_scale_add(
    torch::Tensor input,
    float scaling_factor
) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_sigmoid_scale_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        size
    );
    return output;
}
"""

# The C++ header for the fused function
fused_sigmoid_scale_add_cpp = (
    "torch::Tensor fused_sigmoid_scale_add(torch::Tensor input, float scaling_factor);"
)

# Compile the fused kernel
fused_sigmoid_scale_add = load_inline(
    name="fused_sigmoid_scale_add",
    cuda_sources=fused_sigmoid_scale_add_source,
    cpp_sources=fused_sigmoid_scale_add_cpp,
    functions=["fused_sigmoid_scale_add"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_sigmoid_scale_add = fused_sigmoid_scale_add  # Load the fused kernel

    def forward(self, x):
        gemm_out = self.gemm(x)
        x = self.fused_sigmoid_scale_add.fused_sigmoid_scale_add(gemm_out, self.scaling_factor)
        return x

# Maintaining the original get_inputs and get_init_inputs functions
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    # Generate inputs on CUDA
    return [torch.rand(batch_size, input_size, device='cuda')]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]