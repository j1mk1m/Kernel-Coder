import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA source code for the fused operation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t size,
    float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = input[idx] / divisor;
        output[idx] = (val > 0) ? val : val * 0.01f;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor input, float divisor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_forward", ([&] {
        fused_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size,
            divisor);
    }));

    return output;
}

template <typename scalar_t>
__global__ void fused_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ grad_input,
    int64_t size,
    float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = input[idx] / divisor;
        scalar_t grad_factor = (val > 0) ? 1.0f : 0.01f;
        grad_factor /= divisor;
        grad_input[idx] = grad_output[idx] * grad_factor;
    }
}

torch::Tensor fused_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    float divisor) {
    auto size = input.numel();
    auto grad_input = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "fused_backward", ([&] {
        fused_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            size,
            divisor);
    }));

    return grad_input;
}
"""

cpp_sources = """
extern "C" {
    torch::Tensor fused_forward_cuda(torch::Tensor input, float divisor);
    torch::Tensor fused_backward_cuda(torch::Tensor grad_output, torch::Tensor input, float divisor);
}
"""

# Compile the CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_sources,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda", "fused_backward_cuda"],
    verbose=True
)

class FusedDivLeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, divisor):
        ctx.save_for_backward(input)
        ctx.divisor = divisor
        return fused_ops.fused_forward_cuda(input, divisor)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        divisor = ctx.divisor
        grad_input = fused_ops.fused_backward_cuda(grad_output, input, divisor)
        return grad_input, None

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)
        x = FusedDivLeakyReLU.apply(x, self.divisor)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]