import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hardsigmoid_source = """
#include <torch/extension.h>

__global__ void hardsigmoid_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float tmp = 0.2f * x[idx] + 0.5f;
        out[idx] = fmaxf(0.0f, fminf(1.0f, tmp));
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

hardsigmoid_header = """
torch::Tensor hardsigmoid_cuda(torch::Tensor x);
"""

# Compile the custom CUDA operator
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_header,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid = hardsigmoid  # Reference to the compiled module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid.hardsigmoid_cuda(x)