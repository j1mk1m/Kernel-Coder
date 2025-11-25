import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise operations CUDA kernel
fused_elementwise_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_elementwise_ops_kernel(
    const float* input_data,
    const float* add_value_data,
    float* output_data,
    int batch_size,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int j = idx % out_features;
    float temp = input_data[idx] + add_value_data[j];

    // Swish
    float sigmoid_temp = 1.0f / (1.0f + expf(-temp));
    temp *= sigmoid_temp;

    // Tanh
    temp = tanhf(temp);

    // GELU (exact implementation)
    float x_over_sqrt2 = temp / sqrtf(2.0f);
    float erf_val = erff(x_over_sqrt2);
    temp = 0.5f * temp * (1.0f + erf_val);

    // Hardtanh
    if (temp < -1.0f) temp = -1.0f;
    else if (temp > 1.0f) temp = 1.0f;

    output_data[idx] = temp;
}

torch::Tensor fused_elementwise_ops_cuda(
    torch::Tensor input,
    torch::Tensor add_value
) {
    int batch_size = input.size(0);
    int out_features = input.size(1);

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int num_blocks = (batch_size * out_features + threads_per_block - 1) / threads_per_block;

    fused_elementwise_ops_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        add_value.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

# Define the C++ header for the fused CUDA operator
fused_elementwise_ops_cpp_source = (
    "torch::Tensor fused_elementwise_ops_cuda(torch::Tensor input, torch::Tensor add_value);"
)

# Compile the fused CUDA kernel
fused_elementwise_ops = load_inline(
    name="fused_elementwise_ops",
    cpp_sources=fused_elementwise_ops_cpp_source,
    cuda_sources=fused_elementwise_ops_source,
    functions=["fused_elementwise_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_elementwise_ops = fused_elementwise_ops

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_elementwise_ops.fused_elementwise_ops_cuda(x, self.add_value)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]