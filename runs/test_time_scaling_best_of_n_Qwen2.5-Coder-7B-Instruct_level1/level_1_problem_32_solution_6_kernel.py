import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardTanh activation
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(float* x, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        x[idx] = fmaxf(fminf(val, max_val), min_val);
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val) {
    auto size = x.numel();
    auto out = x.clone();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(out.data_ptr<float>(), size, min_val, max_val);

    return out;
}
"""

hardtanh_cpp_source = (
    "torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val);"
)

# Compile the inline CUDA code for HardTanh activation
hardtanh = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=["hardtanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return hardtanh.hardtanh_cuda(x, -1., 1.)