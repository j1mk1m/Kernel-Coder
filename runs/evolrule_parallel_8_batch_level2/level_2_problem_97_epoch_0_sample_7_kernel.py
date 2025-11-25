import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Swish kernel
fused_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_swish_kernel(
    const float* input, const float bias, const float divide_value,
    float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = (input[idx] + bias) / divide_value;
        float sigmoid_temp = 1.0f / (1.0f + expf(-temp));
        output[idx] = temp * sigmoid_temp;
    }
}

torch::Tensor fused_swish_cuda(torch::Tensor input, float bias, float divide_value) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_swish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias, divide_value,
        output.data_ptr<float>(), size);

    return output;
}
"""

# Compile the fused Swish kernel
fused_swish = load_inline(
    name="fused_swish",
    cpp_sources="",
    cuda_sources=fused_swish_source,
    functions=["fused_swish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_swish = fused_swish

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        # Extract bias scalar value and use fused kernel
        x = self.fused_swish.fused_swish_cuda(
            x, self.bias[0].item(), self.divide_value
        )
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]