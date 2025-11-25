import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA source for ReLU forward and backward
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = max(0.f, x[idx]);
    }
}

torch::Tensor relu_forward_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    relu_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

__global__ void relu_backward_kernel(const float* x, const float* grad_y, float* grad_x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_x[idx] = (x[idx] > 0.f) ? grad_y[idx] : 0.f;
    }
}

torch::Tensor relu_backward_cuda(torch::Tensor x, torch::Tensor grad_y) {
    auto n = x.numel();
    auto grad_x = torch::empty_like(x);
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    relu_backward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), grad_y.data_ptr<float>(), grad_x.data_ptr<float>(), n);
    return grad_x;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor relu_forward_cuda(torch::Tensor x);
torch::Tensor relu_backward_cuda(torch::Tensor x, torch::Tensor grad_y);
"""

# Compile the CUDA code
relu_ops = load_inline(
    name="relu_ops",
    cpp_sources=cpp_source,
    cuda_sources=relu_source,
    functions=["relu_forward_cuda", "relu_backward_cuda"],
    verbose=True
)

class ReLUCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu_ops.relu_forward_cuda(x)

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        return relu_ops.relu_backward_cuda(x, grad_y)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return ReLUCustomFunction.apply(x)

# Parameters and input functions
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []