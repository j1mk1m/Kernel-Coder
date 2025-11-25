import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = std::max(0.0f, std::min(1.0f, val + 3.0f));
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

hardsigmoid_cpp_source = (
    "torch::Tensor hardsigmoid_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for HardSigmoid
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid = hardsigmoid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid.hardsigmoid_cuda(x)