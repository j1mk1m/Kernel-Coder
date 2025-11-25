import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* in_ptr, float* out_ptr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in_ptr[idx];
        out_ptr[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor in) {
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Launch kernel
    relu_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor in);"

# Compile the inline CUDA code for ReLU
relu_module = load_inline(
    name="relu_cuda",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_cuda = relu_module  # Store the CUDA module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.relu_cuda(x)  # Call the custom ReLU kernel