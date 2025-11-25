import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for Sigmoid, scaling, and residual addition
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_sigmoid_scale_add_kernel(
    const float* input_data,
    float scaling_factor,
    float* output_data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input_data[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-x));
        output_data[idx] = x + scaling_factor * sigmoid_val;
    }
}

torch::Tensor fused_sigmoid_scale_add_cuda(
    torch::Tensor input,
    float scaling_factor
) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sigmoid_scale_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scaling_factor,
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

# Header for the fused kernel
fused_ops_header = """
torch::Tensor fused_sigmoid_scale_add_cuda(
    torch::Tensor input,
    float scaling_factor
);
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_header,
    cuda_sources=fused_ops_source,
    functions=["fused_sigmoid_scale_add_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        # Compute Gemm (linear layer)
        gemm_out = self.gemm(x)
        # Apply fused operations: Sigmoid + scaling + residual add
        x = self.fused_ops.fused_sigmoid_scale_add_cuda(gemm_out, self.scaling_factor)
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    # Generate input on CUDA
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]