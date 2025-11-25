import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

class HardsigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input < -3.0, torch.zeros_like(input),
                          torch.where(input > 3.0, torch.ones_like(input),
                                      input * (1.0/6.0) + 0.5))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -3.0] = 0
        grad_input[input >= 3.0] = 0
        grad_input[(input > -3.0) & (input < 3.0)] *= (1.0/6.0)
        return grad_input

# Alternatively, using CUDA kernels for both forward and backward passes for better performance

# Define CUDA kernels
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void HardsigmoidForwardKernel(const scalar_t* __restrict__ input,
                                        scalar_t* __restrict__ output,
                                        const int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        scalar_t x = input[idx];
        if (x < -3.0) {
            output[idx] = 0.0;
        } else if (x > 3.0) {
            output[idx] = 1.0;
        } else {
            output[idx] = x * (1.0 / 6.0) + 0.5;
        }
    }
}

template <typename scalar_t>
__global__ void HardsigmoidBackwardKernel(const scalar_t* __restrict__ grad_output,
                                         const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ grad_input,
                                         const int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        scalar_t x = input[idx];
        if (x <= -3.0 || x >= 3.0) {
            grad_input[idx] = 0.0;
        } else {
            grad_input[idx] = grad_output[idx] * (1.0 / 6.0);
        }
    }
}

torch::Tensor hardsigmoid_forward_cuda(torch::Tensor input) {
    const auto num_elements = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "hardsigmoid_forward_cuda", ([&] {
        HardsigmoidForwardKernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), output.data<scalar_t>(), num_elements);
    }));

    return output;
}

torch::Tensor hardsigmoid_backward_cuda(torch::Tensor grad_output,
                                       torch::Tensor input) {
    const auto num_elements = grad_output.numel();
    auto grad_input = torch::empty_like(grad_output);
    
    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "hardsigmoid_backward_cuda", ([&] {
        HardsigmoidBackwardKernel<scalar_t><<<blocks, threads>>>(
            grad_output.data<scalar_t>(), input.data<scalar_t>(),
            grad_input.data<scalar_t>(), num_elements);
    }));

    return grad_input;
}
"""

# Compile the CUDA kernels
cuda_ext = load_inline(
    name="hardsigmoid_cuda",
    cpp_sources="",
    cuda_sources=hardsigmoid_source,
    functions=[
        "torch::Tensor hardsigmoid_forward_cuda(torch::Tensor input)",
        "torch::Tensor hardsigmoid_backward_cuda(torch::Tensor grad_output, torch::Tensor input)",
    ],
    verbose=True,
)

class HardsigmoidCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return cuda_ext.hardsigmoid_forward_cuda(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return cuda_ext.hardsigmoid_backward_cuda(grad_output, input)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, x):
        # Use the custom CUDA function for forward pass
        return HardsigmoidCudaFunction.apply(x)

# The original get_inputs and get_init_inputs functions remain the same
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()  # Assuming CUDA is available
    return [x]

def get_init_inputs():
    return []