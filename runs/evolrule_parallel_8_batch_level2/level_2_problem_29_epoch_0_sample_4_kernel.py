import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mish_twice_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void mish_twice_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];

        // First Mish
        float exp_val = expf(val);
        float softplus_val = log1pf(exp_val);
        float tanh_soft = tanhf(softplus_val);
        val = val * tanh_soft;

        // Second Mish
        exp_val = expf(val);
        softplus_val = log1pf(exp_val);
        tanh_soft = tanhf(softplus_val);
        val = val * tanh_soft;

        output[idx] = val;
    }
}

torch::Tensor mish_twice_cuda(torch::Tensor input) {
    int size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    mish_twice_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

mish_twice_h = """
torch::Tensor mish_twice_cuda(torch::Tensor input);
"""

mish_twice = load_inline(
    name="mish_twice",
    cpp_sources=mish_twice_h,
    cuda_sources=mish_twice_source,
    functions=["mish_twice_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mish_twice = mish_twice

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.mish_twice.mish_twice_cuda(x)