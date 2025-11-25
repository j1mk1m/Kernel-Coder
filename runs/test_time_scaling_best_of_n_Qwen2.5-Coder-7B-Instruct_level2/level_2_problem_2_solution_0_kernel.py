import torch
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise multiplication
elementwise_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_mul_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

torch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_mul_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_mul_cpp_source = (
    "torch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise multiplication
elementwise_mul = load_inline(
    name="elementwise_mul",
    cpp_sources=elementwise_mul_cpp_source,
    cuda_sources=elementwise_mul_source,
    functions=["elementwise_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Usage in the model
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mul = elementwise_mul

    def forward(self, x, y):
        return self.mul.elementwise_mul_cuda(x, y)