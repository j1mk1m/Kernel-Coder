import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for division and GELU
fused_div_gelu_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void fused_div_gelu_kernel(const float* input, float* output, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float scaled_x = input[idx] / divisor;
        float term = scaled_x * (1.0f + 0.044715f * scaled_x * scaled_x);
        float tanh_val = tanhf(0.79788456f * term);
        output[idx] = 0.5f * scaled_x * (1.0f + tanh_val);
    }
}

torch::Tensor fused_div_gelu_cuda(torch::Tensor input, float divisor) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    fused_div_gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), output.data_ptr<float>(), divisor, size);
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_div_gelu_cuda(torch::Tensor input, float divisor);
"""

# Compile the inline CUDA code for fused division and GELU
fused_div_gelu = load_inline(
    name="fused_div_gelu",
    cpp_sources=cpp_source,
    cuda_sources=fused_div_gelu_source,
    functions=["fused_div_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_div_gelu = fused_div_gelu

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_div_gelu.fused_div_gelu_cuda(x, self.divisor)
        return x

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, output_size, divisor]