from model import Model
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ReLU using vectorized operations
relu_kernel_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vectorized_relu(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < size; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i / 4];
        float4 res = {fmaxf(val.x, 0.f), fmaxf(val.y, 0.f), fmaxf(val.z, 0.f), fmaxf(val.w, 0.f)};
        reinterpret_cast<float4*>(output)[i / 4] = res;
    }
}

torch::Tensor vectorized_relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    vectorized_relu<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}

TORCH_LIBRARY(custom_relu, m) {
  m.def("vectorized_relu", &vectorized_relu_cuda, "Vectorized ReLU");
}
"""

# Inline CUDA compilation
custom_relu = load_inline(
    name="custom_relu",
    cpp_sources="",
    cuda_sources=relu_kernel_code,
    functions=["vectorized_relu_cuda"],
    verbose=True
)

class ModelNew(Model):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_relu.vectorized_relu_cuda(x)