import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the gelu activation function
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 0.5 * x[i] * (1.0 + tanh(sqrt(2.0 / M_PI) * (x[i] + 0.044715 * pow(x[i], 3.0))));
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);
    gelu_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

gelu = load_inline(name="gelu", cpp_sources=gelu_cpp_source, cuda_sources=gelu_source, functions=["gelu_cuda"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        return gelu.gelu_cuda(x)