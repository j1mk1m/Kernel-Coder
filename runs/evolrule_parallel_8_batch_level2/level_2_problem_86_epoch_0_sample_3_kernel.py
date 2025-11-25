import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for division and GELU
fused_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_div_gelu_kernel(const float* input, const float divisor, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] / divisor;
        float a = x + 0.044715f * x * x * x;
        float b = 0.7978845608f * a;
        float tanh_b = tanhf(b);
        output[idx] = 0.5f * x * (1.0f + tanh_b);
    }
}

torch::Tensor fused_div_gelu_cuda(torch::Tensor input, float divisor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_div_gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), divisor, output.data_ptr<float>(), size);
    return output;
}
"""

# The header for the C++ function
fused_div_gelu_cpp_source = """
torch::Tensor fused_div_gelu_cuda(torch::Tensor input, float divisor);
"""

# Compile the CUDA code
fused_div_gelu = load_inline(
    name="fused_div_gelu",
    cpp_sources=fused_div_gelu_cpp_source,
    cuda_sources=fused_div_gelu_source,
    functions=["fused_div_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_div_gelu = fused_div_gelu

    def forward(self, x):
        x = self.linear(x)
        return self.fused_div_gelu.fused_div_gelu_cuda(x, self.divisor)