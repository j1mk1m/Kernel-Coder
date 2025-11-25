import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void hardsigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] + 3.0f;
        val = fmaxf(val, 0.0f);
        val = fminf(val, 6.0f);
        out[idx] = val / 6.0f;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

hardsigmoid_cpp_source = """
torch::Tensor hardsigmoid_cuda(torch::Tensor x);
"""

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
        super().__init__()
        self.hardsigmoid = hardsigmoid

    def forward(self, x):
        return self.hardsigmoid.hardsigmoid_cuda(x)