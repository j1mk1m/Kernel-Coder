import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused element-wise operations
custom_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_elementwise_kernel(
    const float* input,
    float constant,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float min_val = (x < constant) ? x : constant;
        output[idx] = (min_val - constant) * constant;
    }
}

torch::Tensor custom_elementwise_cuda(torch::Tensor input, float constant) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    custom_elementwise_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        constant,
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

custom_elementwise_cpp_source = (
    "torch::Tensor custom_elementwise_cuda(torch::Tensor input, float constant);"
)

# Compile the inline CUDA code
custom_elementwise = load_inline(
    name="custom_elementwise",
    cuda_sources=custom_elementwise_source,
    cpp_sources=custom_elementwise_cpp_source,
    functions=["custom_elementwise_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.custom_elementwise = custom_elementwise

    def forward(self, x):
        # Matrix multiplication with bias (handled by PyTorch's optimized Linear layer)
        x = self.linear(x)
        # Apply fused element-wise operations (min, subtract, multiply) in a single kernel
        x = self.custom_elementwise.custom_elementwise_cuda(x, self.constant.item())
        return x

# Ensure inputs are moved to CUDA as in the original code
batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, constant]