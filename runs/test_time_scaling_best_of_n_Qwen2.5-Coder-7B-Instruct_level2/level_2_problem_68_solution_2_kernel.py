import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min and subtraction
fused_min_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_min_subtract_kernel(const float* x, const float* constant, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] < constant ? x[idx] : constant - x[idx];
    }
}

torch::Tensor fused_min_subtract_cuda(torch::Tensor x, torch::Tensor constant) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_min_subtract_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), constant.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_min_subtract_cpp_source = (
    "torch::Tensor fused_min_subtract_cuda(torch::Tensor x, torch::Tensor constant);"
)

# Compile the inline CUDA code for fused min and subtraction
fused_min_subtract = load_inline(
    name="fused_min_subtract",
    cpp_sources=fused_min_subtract_cpp_source,
    cuda_sources=fused_min_subtract_source,
    functions=["fused_min_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.fused_min_subtract = fused_min_subtract

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_min_subtract.fused_min_subtract_cuda(x, self.constant)
        return x