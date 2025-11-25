import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Gemm_Sigmoid_Scaling_ResidualAdd
gemm_sigmoid_scaling_residualadd_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the Gemm_Sigmoid_Scaling_ResidualAdd operation here using CUDA
// ...

torch::Tensor gemm_sigmoid_scaling_residualadd_cuda(torch::Tensor x, float scaling_factor) {
    // ...
}

"""

gemm_sigmoid_scaling_residualadd_cpp_source = (
    "torch::Tensor gemm_sigmoid_scaling_residualadd_cuda(torch::Tensor x, float scaling_factor);"
)

# Compile the inline CUDA code for Gemm_Sigmoid_Scaling_ResidualAdd
gemm_sigmoid_scaling_residualadd = load_inline(
    name="gemm_sigmoid_scaling_residualadd",
    cpp_sources=gemm_sigmoid_scaling_residualadd_cpp_source,
    cuda_sources=gemm_sigmoid_scaling_residualadd_source,
    functions=["gemm_sigmoid_scaling_residualadd_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        x = gemm_sigmoid_scaling_residualadd_cuda(x, self.scaling_factor)
        return x

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]