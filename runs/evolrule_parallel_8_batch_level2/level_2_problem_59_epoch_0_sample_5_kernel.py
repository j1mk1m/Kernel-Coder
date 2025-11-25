import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Swish + scaling kernels
fused_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_swish_scale_forward(const float* y_in, float* out, float scaling, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float yi = y_in[idx];
        float exp_neg_yi = expf(-yi);
        float s = 1.0f / (1.0f + exp_neg_yi);
        out[idx] = yi * s * scaling;
    }
}

__global__ void fused_swish_scale_backward(const float* y_in, const float* grad_out, float scaling, float* grad_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = y_in[idx];
        float exp_neg_y = expf(-y);
        float s = 1.0f / (1.0f + exp_neg_y);
        float term = s + y * s * (1.0f - s);
        grad_in[idx] = grad_out[idx] * scaling * term;
    }
}

torch::Tensor fused_swish_scale_forward_cuda(torch::Tensor y_in, float scaling) {
    auto size = y_in.numel();
    auto out = torch::empty_like(y_in);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_swish_scale_forward<<<num_blocks, block_size>>>(y_in.data_ptr<float>(), out.data_ptr<float>(), scaling, size);
    return out;
}

torch::Tensor fused_swish_scale_backward_cuda(torch::Tensor y_in, torch::Tensor grad_out, float scaling) {
    auto size = y_in.numel();
    auto grad_in = torch::empty_like(y_in);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_swish_scale_backward<<<num_blocks, block_size>>>(y_in.data_ptr<float>(), grad_out.data_ptr<float>(), scaling, grad_in.data_ptr<float>(), size);
    return grad_in;
}
"""

fused_swish_scale_cpp_source = """
torch::Tensor fused_swish_scale_forward_cuda(torch::Tensor y_in, float scaling);
torch::Tensor fused_swish_scale_backward_cuda(torch::Tensor y_in, torch::Tensor grad_out, float scaling);
"""

fused_swish_scale = load_inline(
    name="fused_swish_scale",
    cpp_sources=[fused_swish_scale_cpp_source],
    cuda_sources=[fused_swish_scale_source],
    functions=["fused_swish_scale_forward_cuda", "fused_swish_scale_backward_cuda"],
    verbose=True
)

class FusedSwishScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, scaling):
        ctx.save_for_backward(y)
        ctx.scaling = scaling
        return fused_swish_scale.fused_swish_scale_forward_cuda(y, scaling)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        scaling = ctx.scaling
        return fused_swish_scale.fused_swish_scale_backward_cuda(y, grad_output, scaling), None

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        y = self.matmul(x)
        return FusedSwishScale.apply(y, self.scaling_factor)

# The get_inputs and get_init_inputs are same as original?
# Assuming they are defined as in the original problem, so the user's code includes them, but in the new code, they are same.

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]