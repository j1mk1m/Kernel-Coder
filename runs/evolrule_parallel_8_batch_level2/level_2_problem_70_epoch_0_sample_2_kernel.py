import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_sigmoid_scale_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_sigmoid_scale_add_kernel(
    const float* input, float scaling_factor, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        float scaled_sigmoid = scaling_factor * sigmoid_val;
        output[idx] = val + scaled_sigmoid;
    }
}

torch::Tensor fused_sigmoid_scale_add_cuda(
    torch::Tensor input, float scaling_factor) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_sigmoid_scale_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), scaling_factor, output.data_ptr<float>(), size
    );

    return output;
}
"""

fused_sigmoid_scale_add_cpp_source = """
torch::Tensor fused_sigmoid_scale_add_cuda(
    torch::Tensor input, float scaling_factor);
"""

# Compile the CUDA code for fused operation
fused_sigmoid_scale_add = load_inline(
    name="fused_sigmoid_scale_add",
    cpp_sources=fused_sigmoid_scale_add_cpp_source,
    cuda_sources=fused_sigmoid_scale_add_source,
    functions=["fused_sigmoid_scale_add_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.gemm(x)
        x = fused_sigmoid_scale_add.fused_sigmoid_scale_add_cuda(x, self.scaling_factor)
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]