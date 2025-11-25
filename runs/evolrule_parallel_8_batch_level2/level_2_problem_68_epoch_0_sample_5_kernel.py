import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for min and subtraction
fused_min_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_min_subtract_kernel(const float* x, const float c, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (x[idx] < c) ? (x[idx] - c) : 0.0f;
    }
}

torch::Tensor fused_min_subtract_cuda(torch::Tensor x, float c) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_min_subtract_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), c, out.data_ptr<float>(), size);

    return out;
}
"""

fused_min_subtract_header = "torch::Tensor fused_min_subtract_cuda(torch::Tensor x, float c);"

# Compile the fused kernel
fused_min_subtract = load_inline(
    name="fused_min_subtract",
    cpp_sources=fused_min_subtract_header,
    cuda_sources=fused_min_subtract_source,
    functions=["fused_min_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.fused_min_subtract = fused_min_subtract  # Reference to the compiled kernel

    def forward(self, x):
        x = self.linear(x)
        c = self.constant.item()
        x = self.fused_min_subtract.fused_min_subtract_cuda(x, c)
        return x