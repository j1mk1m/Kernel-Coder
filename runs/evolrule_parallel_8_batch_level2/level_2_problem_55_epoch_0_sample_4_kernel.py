import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum of max pairs and scaling
sum_max_pairs_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_max_pairs_kernel(
    const float* x,
    float* out,
    int batch_size,
    int out_features,
    float scale_factor
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    float sum = 0.0;
    for (int i = 0; i < (out_features / 2); ++i) {
        const int idx = 2 * i;
        const float a = x[batch_idx * out_features + idx];
        const float b = x[batch_idx * out_features + idx + 1];
        sum += max(a, b);
    }
    out[batch_idx] = sum * scale_factor;
}

torch::Tensor sum_max_pairs_cuda(torch::Tensor x, float scale_factor) {
    const int batch_size = x.size(0);
    const int out_features = x.size(1);
    const int threads_per_block = 128;
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    auto out = torch::empty({batch_size}, x.options());

    sum_max_pairs_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features,
        scale_factor
    );

    return out;
}
"""

sum_max_pairs_cpp_source = "torch::Tensor sum_max_pairs_cuda(torch::Tensor x, float scale_factor);"

# Compile the inline CUDA code
sum_max_pairs = load_inline(
    name="sum_max_pairs",
    cpp_sources=sum_max_pairs_cpp_source,
    cuda_sources=sum_max_pairs_source,
    functions=["sum_max_pairs_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.sum_max_pairs = sum_max_pairs  # Store the kernel module

    def forward(self, x):
        x = self.matmul(x)
        return self.sum_max_pairs.sum_max_pairs_cuda(x, self.scale_factor)

# Input functions (ensuring CUDA tensors)
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]