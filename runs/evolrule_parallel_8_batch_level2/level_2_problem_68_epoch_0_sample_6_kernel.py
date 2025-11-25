import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* x, const float C, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = (val < C) ? (val - C) : 0.0f;
    }
}

torch::Tensor fused_kernel_forward(torch::Tensor x, float C) {
    auto output = torch::empty_like(x);
    int size = x.numel();

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), C, output.data_ptr<float>(), size);

    return output;
}
"""

# Compile the fused kernel
fused_kernel = load_inline(
    name="fused_kernel",
    cuda_sources=fused_kernel_source,
    functions=["fused_kernel_forward"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        x = self.linear(x)
        c = self.constant.item()
        x = fused_kernel.fused_kernel_forward(x, c)
        return x

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, constant]