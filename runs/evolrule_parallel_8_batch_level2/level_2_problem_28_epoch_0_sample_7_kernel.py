import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused element-wise addition and multiplication kernel
fusion_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_addmul_kernel(const scalar_t* x, const scalar_t* y, scalar_t* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t a = x[idx];
        scalar_t b = y[idx];
        out[idx] = (a + b) * b; // (x + y) * y
    }
}

torch::Tensor fused_addmul_cuda(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "fused_addmul_cuda", ([&] {
        fused_addmul_kernel<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), size);
    }));
    
    return out;
}
"""

fusion_cpp = "torch::Tensor fused_addmul_cuda(torch::Tensor x, torch::Tensor y);"

# Compile fused kernel
fused_addmul = load_inline(
    name="fused_addmul",
    cpp_sources=fusion_cpp,
    cuda_sources=fusion_kernel,
    functions=["fused_addmul_cuda"],
    verbose=True
)

# Define optimized instance normalization kernel
# This is a simplified version assuming channel-wise computation
# Actual implementation would need to handle mean/variance computation
# However, for the sake of optimization, we'll keep using PyTorch's optimized version
# Note: InstanceNorm2d is already highly optimized in PyTorch, so not replacing it

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.fused_addmul = fused_addmul

    def forward(self, x, y):
        x = self.bmm(x)
        # Keep instance norm as is (already optimized)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        # Replace add and multiply with fused kernel
        return self.fused_addmul.fused_addmul_cuda(x, y)

def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    return [torch.rand(batch_size, in_features).cuda(), torch.rand(batch_size, out_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]