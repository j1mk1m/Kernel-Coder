import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float abs_xi = fabsf(xi);
        float exp_val = expf(-abs_xi);
        out[idx] = (xi > 0) ? (xi + log1pf(exp_val)) : log1pf(exp_val);
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    int size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus_cuda.softplus_cuda(x.cuda())

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []