import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation
selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

selu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define SCALE 1.05070098f
#define ALPHA 1.67326324f

__global__ void selu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        if (xi > 0.0f) {
            out[idx] = SCALE * xi;
        } else {
            out[idx] = SCALE * ALPHA * (expf(xi) - 1.0f);
        }
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    selu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

# Compile the inline CUDA code for SELU
selu_extension = load_inline(
    name="selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_cuda_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.selu_cuda = selu_extension.selu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda(x)