import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel for bias addition and ReLU
fused_add_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_relu_kernel(const float* x, const float* bias, float* out, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int j = idx % out_features;
        out[idx] = fmaxf(x[idx] + bias[j], 0.0f);
    }
}

torch::Tensor fused_add_relu_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_add_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features
    );

    return out;
}
"""

fused_add_relu_cpp_source = "torch::Tensor fused_add_relu_cuda(torch::Tensor x, torch::Tensor bias);"

# Compile the fused kernel
fused_add_relu = load_inline(
    name="fused_add_relu",
    cpp_sources=fused_add_relu_cpp_source,
    cuda_sources=fused_add_relu_source,
    functions=["fused_add_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_relu = fused_add_relu  # Reference to the compiled module

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_add_relu.fused_add_relu_cuda(x, self.bias)
        return x

# These functions are unchanged from the original
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bias_shape]