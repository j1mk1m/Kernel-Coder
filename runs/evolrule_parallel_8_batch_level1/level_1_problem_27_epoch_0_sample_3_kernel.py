import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void selu_kernel(const float* input, float* output, int n) {
    const float lambda = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float out;
        if (x >= 0.0f) {
            out = lambda * x;
        } else {
            out = lambda * alpha * (exp(-x) - 1.0f);
        }
        output[idx] = out;
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int grid_size = (input.numel() + block_size - 1) / block_size;

    selu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());

    return output;
}
"""

selu_cpp_source = """
torch::Tensor selu_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for SELU
selu = load_inline(
    name="selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.selu_cuda = selu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x)