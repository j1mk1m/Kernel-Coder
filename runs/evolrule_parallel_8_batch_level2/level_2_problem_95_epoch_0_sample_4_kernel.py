import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activations_kernel(
    const float* input,
    const float* add_value,
    float* output,
    int total_elements,
    int add_value_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = input[idx];
    int feature_idx = idx % add_value_size;
    x += add_value[feature_idx];

    // Swish
    x = x * (1.0f / (1.0f + expf(-x)));

    // Tanh
    x = tanhf(x);

    // GELU approximation
    const float sqrt_2_over_pi = 0.7978845608f;
    const float a = 0.044715f;
    float inner = sqrt_2_over_pi * (x + a * x*x*x);
    x = 0.5f * x * (1.0f + tanhf(inner));

    // Hardtanh
    if (x < -1.0f) x = -1.0f;
    else if (x > 1.0f) x = 1.0f;

    output[idx] = x;
}

torch::Tensor fused_activations_cuda(
    torch::Tensor input,
    torch::Tensor add_value) {
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    int add_value_size = add_value.size(0);

    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_activations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        add_value.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        add_value_size
    );

    return output;
}
"""

# Header for the C++ function
fused_ops_header = """
torch::Tensor fused_activations_cuda(
    torch::Tensor input,
    torch::Tensor add_value);
"""

# Load the CUDA extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_header,
    cuda_sources=fused_ops_source,
    functions=["fused_activations_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        x = self.matmul(x)
        x = fused_ops.fused_activations_cuda(x, self.add_value)
        return x

def get_inputs():
    return [torch.rand(1024, 8192)]

def get_init_inputs():
    return [8192, 8192, (8192,)]