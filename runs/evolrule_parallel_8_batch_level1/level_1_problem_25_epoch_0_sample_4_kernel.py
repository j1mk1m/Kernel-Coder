import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_CHECK()                                                           \\
    do {                                                                               \\
        cudaError_t __err__ = cudaGetLastError();                                      \\
        if (__err__ != cudaSuccess) {                                                  \\
            fprintf(stderr, "CUDA error: %s:%d: %s\\n", __FILE__, __LINE__,             \\
                    cudaGetErrorString(__err__));                                      \\
            abort();                                                                   \\
        }                                                                               \\
    } while (0)

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sigmoid_xi = 1.0f / (1.0f + expf(-xi));
        out[idx] = xi * sigmoid_xi;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    CUDA_KERNEL_CHECK();

    return out;
}
"""

cpp_sources = "torch::Tensor swish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish activation
swish = load_inline(
    name="swish",
    cpp_sources=cpp_sources,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = swish  # Stores the compiled CUDA module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish.swish_cuda(x)  # Calls the custom CUDA kernel