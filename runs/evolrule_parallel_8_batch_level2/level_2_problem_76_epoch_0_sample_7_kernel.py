import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for add and ReLU
fused_add_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_relu(const float* x, const float* bias, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int j = idx % out_features;
        float val = x[idx] + bias[j];
        out[idx] = fmaxf(val, 0.0f);  // Using fmaxf for ReLU
    }
}

torch::Tensor fused_add_relu_cuda(torch::Tensor x, torch::Tensor bias) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    int bias_size = bias.size(0);
    if (out_features != bias_size) {
        throw std::runtime_error("Bias size must match output features");
    }

    auto out = torch::empty_like(x);

    const int size = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_relu<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, out_features);

    return out;
}
"""

# Define the C++ header for the fused kernel
fused_add_relu_cpp_source = (
    "torch::Tensor fused_add_relu_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the fused CUDA kernel
fused_add_relu = load_inline(
    name="fused_add_relu",
    cpp_sources=fused_add_relu_cpp_source,
    cuda_sources=fused_add_relu_source,
    functions=["fused_add_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_relu = fused_add_relu.fused_add_relu_cuda

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_add_relu(x, self.bias)
        return x