import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for tanh
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    input = input.contiguous();  // Ensure input is contiguous
    int64_t n = input.numel();
    auto output = torch::empty_like(input);  // Output has same strides as contiguous input
    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

tanh_cpp_source = """
torch::Tensor tanh_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
tanh = load_inline(
    name="tanh",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh_cuda = tanh  # Store the loaded CUDA function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_cuda.tanh_cuda(x)