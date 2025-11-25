import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus with threshold
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Threshold value as per the problem statement
#define THRESHOLD 20.0f

__global__ void softplus_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val >= THRESHOLD) {
            out[idx] = val;
        } else {
            out[idx] = logf(1.0f + expf(val));
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

softplus_cpp_source = (
    "torch::Tensor softplus_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Softplus
softplus = load_inline(
    name="softplus",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus = softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA device
        if x.is_cuda:
            return self.softplus.softplus_cuda(x)
        else:
            # Fallback to PyTorch's implementation if input is on CPU
            # Though in this problem setup, inputs are expected to be on CUDA
            return torch.nn.functional.softplus(x)

def get_inputs():
    # Generate inputs on CUDA device
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []