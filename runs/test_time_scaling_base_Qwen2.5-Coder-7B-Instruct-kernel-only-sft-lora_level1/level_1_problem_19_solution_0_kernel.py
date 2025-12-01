import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

relu_cpp_source = (
    "torch::Tensor relu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for ReLU activation
relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu = relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_cuda(x)