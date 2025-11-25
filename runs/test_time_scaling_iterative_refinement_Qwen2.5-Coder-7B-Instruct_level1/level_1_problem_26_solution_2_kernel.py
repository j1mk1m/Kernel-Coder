import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float mul = 0.5f * val * (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * val * val)));
        x[idx] = mul;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::clone(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for GELU
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu.gelu_cuda(x)