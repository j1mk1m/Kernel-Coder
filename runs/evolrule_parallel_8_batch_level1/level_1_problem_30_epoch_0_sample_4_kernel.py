import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for Softsign forward and backward
softsign_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_forward_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float denom = 1.0f + fabsf(xi);
        y[i] = xi / denom;
    }
}

__global__ void softsign_backward_kernel(const float* x, const float* grad_out, float* grad_x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float denom = 1.0f + fabsf(xi);
        float denom_sq = denom * denom;
        grad_x[i] = grad_out[i] / denom_sq;
    }
}

torch::Tensor softsign_forward(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    softsign_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor softsign_backward(torch::Tensor x, torch::Tensor grad_out) {
    auto n = x.numel();
    auto grad_x = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    softsign_backward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), grad_out.data_ptr<float>(), grad_x.data_ptr<float>(), n);
    return grad_x;
}
"""

# C++ headers for the CUDA functions
softsign_cuda_header = """
torch::Tensor softsign_forward(torch::Tensor x);
torch::Tensor softsign_backward(torch::Tensor x, torch::Tensor grad_out);
"""

# Load the CUDA extensions
softsign_cuda = load_inline(
    name='softsign_cuda',
    cpp_sources=softsign_cuda_header,
    cuda_sources=softsign_cuda_source,
    functions=['softsign_forward', 'softsign_backward'],
    verbose=True
)

# Define the custom autograd function
class SoftsignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_contig = x.contiguous()
        ctx.save_for_backward(x_contig)
        return softsign_cuda.softsign_forward(x_contig)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_out_contig = grad_output.contiguous()
        return softsign_cuda.softsign_backward(x, grad_out_contig)

# Define the optimized model
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SoftsignFunction.apply(x)

# Original helper functions (unchanged)
def get_inputs():
    x = torch.rand(batch_size, dim).cuda()  # Assuming CUDA is available
    return [x]

def get_init_inputs():
    return []