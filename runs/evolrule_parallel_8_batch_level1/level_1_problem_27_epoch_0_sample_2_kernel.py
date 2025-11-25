import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 4096
dim = 393216

# Define the custom CUDA kernel for SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void selu_kernel(const float* x, float* out, int size) {
    const float scale = 1.0507009873554805f;
    const float alpha = 1.6732632423543762f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float max_val = fmaxf(xi, 0.0f);
        float min_val = fminf(xi, 0.0f);
        float term1 = max_val;
        float term2 = alpha * (expf(min_val) - 1.0f);
        out[idx] = scale * (term1 + term2);
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    selu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for SELU
selu = load_inline(
    name="selu",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True,
    extra_cuda_cflags=["-use_fast_math"],  # Enable fast math optimizations
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.selu = selu  # Store the loaded CUDA function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu.selu_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()  # Ensure input is on CUDA
    return [x]

def get_init_inputs():
    return []