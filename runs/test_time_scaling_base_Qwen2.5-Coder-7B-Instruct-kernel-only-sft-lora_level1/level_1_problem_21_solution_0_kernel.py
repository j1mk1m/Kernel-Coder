import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Sigmoid activation
sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to calculate exp(-x) safely
__device__ float exp_minus_x(float x) {
    if (x > 0.0f) {
        return 1.0f / (1.0f + exp(-x));
    } else {
        return exp(x);
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = exp_minus_x(input[idx]);
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

sigmoid_cpp_source = (
    "torch::Tensor sigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Sigmoid activation
sigmoid = load_inline(
    name="sigmoid",
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=["sigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid.sigmoid_cuda(x)