import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* x, float* y, int batch_size, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_features) {
        return;
    }

    int row = idx / num_features;
    int col = idx % num_features;

    float max_val = -1e9;
    for (int i = 0; i < num_features; ++i) {
        if (x[row * num_features + i] > max_val) {
            max_val = x[row * num_features + i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < num_features; ++i) {
        sum_exp += exp(x[row * num_features + i] - max_val);
    }

    y[idx] = exp(x[row * num_features + col] - max_val) / sum_exp;
}
"""

softmax_cpp_source = (
    "void softmax_kernel(const float* x, float* y, int batch_size, int num_features);"
)

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_kernel"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features = x.shape
        y = torch.zeros_like(x)
        softmax_kernel<<<batch_size, num_features>>>(x.data_ptr(), y.data_ptr(), batch_size, num_features)
        return y