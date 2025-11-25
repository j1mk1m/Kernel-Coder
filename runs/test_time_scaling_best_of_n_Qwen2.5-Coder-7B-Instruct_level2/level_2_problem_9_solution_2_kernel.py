import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the custom matrix multiplication kernel here
// ...

torch::Tensor matrix_mul_cuda(torch::Tensor a, torch::Tensor b) {
    // ...
    return out;
}
"""

matrix_mul_cpp_source = (
    "torch::Tensor matrix_mul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication
matrix_mul = load_inline(
    name="matrix_mul",
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=["matrix_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.matrix_mul = matrix_mul

    def forward(self, x):
        x = self.linear(x)
        x = self.matrix_mul(x, torch.tensor([self.subtract_value]).repeat(x.size(0)).unsqueeze(-1).cuda())
        x = self.matrix_mul(x, torch.tensor([self.multiply_value]).repeat(x.size(0)).unsqueeze(-1).cuda())
        x = torch.relu(x)
        return x