import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 4096
dim = 393216

softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        const float threshold = 20.0f;
        if (xi > threshold) {
            out[idx] = xi;
        } else if (xi < -threshold) {
            out[idx] = expf(xi);
        } else {
            if (xi >= 0) {
                out[idx] = xi + logf(1.0f + expf(-xi));
            } else {
                out[idx] = logf(1.0f + expf(xi));
            }
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    int size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor x);
"""

# Compile the CUDA extension
softplus = load_inline(
    name="softplus",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus_cuda = softplus

    def forward(self, x):
        return self.softplus_cuda.softplus_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []