import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused add-multiply CUDA kernel
fused_add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_mul_kernel(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (x[idx] + y[idx]) * y[idx];
    }
}

torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    assert(x.sizes() == y.sizes());
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_mul_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_add_mul_header = "torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y);"

# Compile the fused add-multiply kernel
fused_add_mul = load_inline(
    name="fused_add_mul",
    cpp_sources=fused_add_mul_header,
    cuda_sources=fused_add_mul_source,
    functions=["fused_add_mul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.fused_add_mul = fused_add_mul

    def forward(self, x, y):
        x = self.bmm(x)
        # Apply instance normalization on reshaped tensor
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        # Use fused kernel for element-wise operations
        x = self.fused_add_mul.fused_add_mul_cuda(x, y)
        return x