import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min operation
min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_kernel(const float* x, const float* constant, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fminf(x[idx], constant[idx]);
    }
}

torch::Tensor min_cuda(torch::Tensor x, torch::Tensor constant) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    min_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), constant.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

min_cpp_source = (
    "torch::Tensor min_cuda(torch::Tensor x, torch::Tensor constant);"
)

# Compile the inline CUDA code for min operation
min_op = load_inline(
    name="min_op",
    cpp_sources=min_cpp_source,
    cuda_sources=min_source,
    functions=["min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for subtraction operation
subtraction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtraction_kernel(const float* x, const float* constant, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] - constant[idx];
    }
}

torch::Tensor subtraction_cuda(torch::Tensor x, torch::Tensor constant) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtraction_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), constant.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

subtraction_cpp_source = (
    "torch::Tensor subtraction_cuda(torch::Tensor x, torch::Tensor constant);"
)

# Compile the inline CUDA code for subtraction operation
subtraction_op = load_inline(
    name="subtraction_op",
    cpp_sources=subtraction_cpp_source,
    cuda_sources=subtraction_source,
    functions=["subtraction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.min_op = min_op
        self.subtraction_op = subtraction_op

    def forward(self, x):
        x = self.linear(x)
        x = self.min_op.min_cuda(x, self.constant.expand_as(x))
        x = self.subtraction_op.subtraction_cuda(x, self.constant.expand_as(x))
        return x