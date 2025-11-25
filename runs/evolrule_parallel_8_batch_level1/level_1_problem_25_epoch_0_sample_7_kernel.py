import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_forward_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sigmoid_xi = 1.0f / (1.0f + expf(-xi));
        y[idx] = xi * sigmoid_xi;
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

swish_cpp_source = "torch::Tensor swish_forward_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish activation
swish_ops = load_inline(
    name="swish_ops",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_forward = swish_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_forward.swish_forward_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []