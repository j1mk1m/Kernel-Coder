import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.autograd import Function

# Define the fused Mish + Tanh CUDA kernels
fused_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_mish_tanh_forward(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    scalar_t x = input[idx];
    scalar_t softplus_x = log(1 + exp(x));
    scalar_t mish_out = x * tanh(softplus_x);
    scalar_t tanh_out = tanh(mish_out);
    output[idx] = tanh_out;
}

template <typename scalar_t>
__global__ void fused_mish_tanh_backward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    int64_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    scalar_t x = input[idx];
    scalar_t go = grad_output[idx];

    scalar_t softplus_x = log(1 + exp(x));
    scalar_t mish_out = x * tanh(softplus_x);
    scalar_t z = tanh(mish_out);

    scalar_t dz_dy = 1 - z * z;
    scalar_t tanh_softplus = tanh(softplus_x);
    scalar_t sech2 = 1 - tanh_softplus * tanh_softplus;
    scalar_t sigmoid_x = 1.0 / (1 + exp(-x));
    scalar_t dy_dx = tanh_softplus + x * sech2 * sigmoid_x;

    grad_input[idx] = go * dz_dy * dy_dx;
}

at::Tensor fused_mish_tanh_forward_cuda(at::Tensor input) {
    auto output = at::empty_like(input);
    int threads = 1024;
    int blocks = (input.numel() + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_mish_tanh_forward", ([&] {
        fused_mish_tanh_forward<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel());
    }));
    return output;
}

at::Tensor fused_mish_tanh_backward_cuda(
    at::Tensor input,
    at::Tensor grad_output) {
    auto grad_input = at::zeros_like(input);
    int threads = 1024;
    int blocks = (input.numel() + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_mish_tanh_backward", ([&] {
        fused_mish_tanh_backward<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            input.numel());
    }));
    return grad_input;
}
"""

# Compile the CUDA code
fused_mish_tanh_cuda = load_inline(
    name="fused_mish_tanh",
    cuda_sources=fused_mish_tanh_source,
    extra_cuda_cflags=['-lineinfo'],
    verbose=True
)

class FusedMishTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return fused_mish_tanh_cuda.fused_mish_tanh_forward_cuda(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = fused_mish_tanh_cuda.fused_mish_tanh_backward_cuda(input, grad_output)
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = FusedMishTanhFunction.apply(x)
        return x