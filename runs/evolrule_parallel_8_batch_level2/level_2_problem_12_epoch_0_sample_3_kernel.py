import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

scaling_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaling_leaky_relu_kernel(
    const float* input, 
    float* output,
    float multiplier,
    float negative_slope,
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] * multiplier;
        output[idx] = (val >= 0) ? val : val * negative_slope;
    }
}

torch::Tensor scaling_leaky_relu_cuda(
    torch::Tensor input,
    float multiplier,
    float negative_slope) 
{
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scaling_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        multiplier,
        negative_slope,
        size);

    return output;
}
"""

scaling_leaky_relu_cpp_header = (
    "torch::Tensor scaling_leaky_relu_cuda(torch::Tensor input, float multiplier, float negative_slope);"
)

scaling_leaky_relu = load_inline(
    name="scaling_leaky_relu",
    cpp_sources=scaling_leaky_relu_cpp_header,
    cuda_sources=scaling_leaky_relu_source,
    functions=["scaling_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.scaling_leaky_relu = scaling_leaky_relu

    def forward(self, x):
        x = self.gemm(x)
        return self.scaling_leaky_relu.scaling_leaky_relu_cuda(x, self.multiplier, self.negative_slope)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]