import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom fused bias and ReLU kernel
fused_bias_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_relu_kernel(const float* input, const float* bias, float* output, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int j = idx % out_features;
        float val = input[idx] + bias[j];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor fused_bias_relu_cuda(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int num_elements = batch_size * out_features;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_bias_relu_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

# Load the fused kernel
fused_bias_relu = load_inline(
    name="fused_bias_relu",
    cuda_sources=fused_bias_relu_source,
    functions=["fused_bias_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused = fused_bias_relu  # Store the loaded module as an attribute

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused.fused_bias_relu_cuda(x, self.bias)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bias_shape]