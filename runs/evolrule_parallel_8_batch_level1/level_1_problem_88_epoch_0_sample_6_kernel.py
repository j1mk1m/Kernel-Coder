import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GELU forward and backward pass
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void gelu_forward_backward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    scalar_t* __restrict__ dy,
    scalar_t* __restrict__ dx,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    scalar_t xi = x[idx];
    scalar_t x_cubed = xi * xi * xi;
    scalar_t inner = sqrt(2.0f / M_PI) * (xi + 0.044715f * x_cubed);
    scalar_t tanh_inner = tanh(inner);
    scalar_t gelu_val = 0.5f * xi * (1.0f + tanh_inner);

    y[idx] = gelu_val;

    // Backward pass
    if (dy != nullptr && dx != nullptr) {
        scalar_t dgelu = dy[idx];
        scalar_t dgelu_dx = 0.5f * (1.0f + tanh_inner) + 0.5f * xi * (sqrt(2.0f / M_PI) * (1 + 0.134145f * x_cubed) * (1 - tanh_inner * tanh_inner));
        dx[idx] = dgelu * dgelu_dx;
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    auto y = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (x.numel() + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "gelu_forward", ([&] {
        gelu_forward_backward_kernel<scalar_t><<<num_blocks, block_size>>>(
            x.data<scalar_t>(),
            y.data<scalar_t>(),
            nullptr,
            nullptr,
            x.numel()
        );
    }));

    return y;
}

std::tuple<torch::Tensor, torch::Tensor> gelu_forward_backward(torch::Tensor x, torch::Tensor dy) {
    auto y = torch::empty_like(x);
    auto dx = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (x.numel() + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "gelu_forward_backward", ([&] {
        gelu_forward_backward_kernel<scalar_t><<<num_blocks, block_size>>>(
            x.data<scalar_t>(),
            y.data<scalar_t>(),
            dy.data<scalar_t>(),
            dx.data<scalar_t>(),
            x.numel()
        );
    }));

    return std::make_tuple(y, dx);
}

"""

# Load the CUDA kernels
gelu_ops = load_inline(
    name="gelu_ops",
    cpp_sources="",
    cuda_sources=gelu_cuda_source,
    functions=["gelu_forward", "gelu_forward_backward"],
    verbose=False,
    extra_cflags=["-g", "-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return gelu_ops.gelu_forward(x)
    
    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        y, dx = gelu_ops.gelu_forward_backward(x, dy)
        return dx

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return ModelNew.apply(x)

def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda', requires_grad=True)]

def get_init_inputs():
    return []