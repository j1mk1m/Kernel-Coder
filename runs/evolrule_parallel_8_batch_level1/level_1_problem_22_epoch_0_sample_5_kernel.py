import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Tanh activation
tanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* a, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(a[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor a);"

# Compile the inline CUDA code for Tanh
tanh = load_inline(
    name="tanh",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_cuda_source,
    functions=["tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = tanh  # Store the CUDA extension module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh.tanh_cuda(x)