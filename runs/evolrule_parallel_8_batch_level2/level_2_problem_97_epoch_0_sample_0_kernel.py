import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
extern "C" {
    torch::Tensor fused_elementwise_cuda(torch::Tensor input, float bias, float divide_value);
}
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_elementwise_kernel(
    const float* input,
    float bias,
    float divide_value,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = (input[idx] + bias) / divide_value;
        float sigmoid_temp = 1.0f / (1.0f + expf(-temp));
        output[idx] = temp * sigmoid_temp;
    }
}

torch::Tensor fused_elementwise_cuda(
    torch::Tensor input,
    float bias,
    float divide_value
) {
    int size = input.numel();
    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    fused_elementwise_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        bias,
        divide_value,
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

# Load the fused elementwise kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_elementwise_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_elementwise = fused_elementwise

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        bias_val = self.bias.item()
        divide_val = self.divide_value
        x = self.fused_elementwise.fused_elementwise_cuda(x, bias_val, divide_val)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]