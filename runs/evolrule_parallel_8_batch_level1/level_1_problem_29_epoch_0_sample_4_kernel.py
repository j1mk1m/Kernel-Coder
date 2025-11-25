import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus approximation
softplus_approx_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_approx_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        if (xi > 20.0f) {  // Use threshold 20 for numerical stability
            out[idx] = xi;
        } else {
            out[idx] = logf(1.0f + expf(xi));
        }
    }
}

torch::Tensor softplus_approx_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_approx_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size
    );

    return out;
}
"""

softplus_approx_cpp_source = "torch::Tensor softplus_approx_cuda(torch::Tensor x);"

# Compile the inline CUDA code
softplus_approx = load_inline(
    name="softplus_approx",
    cpp_sources=softplus_approx_cpp_source,
    cuda_sources=softplus_approx_source,
    functions=["softplus_approx_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus_approx = softplus_approx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus_approx.softplus_approx_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()  # Ensure inputs are on CUDA
    return [x]

def get_init_inputs():
    return []