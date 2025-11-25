import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernels for forward and backward passes
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Forward kernel: optimized Softplus computation
__global__ void softplus_forward(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float abs_x = fabs(xi);
        float term = fmaxf(xi, 0.0f);
        float exp_term = expf(-abs_x);
        out[idx] = term + logf(1.0f + exp_term);
    }
}

// Backward kernel: computes the gradient (sigmoid function)
__global__ void softplus_backward(const float* x, const float* grad_out, float* grad_x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float exp_neg_x = expf(-xi);
        float sigmoid_val = 1.0f / (1.0f + exp_neg_x);
        grad_x[idx] = grad_out[idx] * sigmoid_val;
    }
}

// Wrapper functions for forward and backward passes
torch::Tensor softplus_forward_cuda(torch::Tensor x) {
    int64_t n = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    softplus_forward<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

torch::Tensor softplus_backward_cuda(torch::Tensor x, torch::Tensor grad_out) {
    int64_t n = x.numel();
    auto grad_x = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    softplus_backward<<<num_blocks, block_size>>>(x.data_ptr<float>(), grad_out.data_ptr<float>(), grad_x.data_ptr<float>(), n);
    return grad_x;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_forward_cuda(torch::Tensor x);
torch::Tensor softplus_backward_cuda(torch::Tensor x, torch::Tensor grad_out);
"""

# Compile the CUDA code inline
softplus = load_inline(
    name="softplus_kernels",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_forward_cuda", "softplus_backward_cuda"],
    verbose=True,
)

# Define the autograd function to integrate with PyTorch
class SoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return softplus.softplus_forward_cuda(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return softplus.softplus_backward_cuda(x, grad_output)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SoftplusFunction.apply(x)