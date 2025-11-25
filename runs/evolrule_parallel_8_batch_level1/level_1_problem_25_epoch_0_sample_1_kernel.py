import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernels for Swish activation and its backward pass
swish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigma = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigma;
    }
}

torch::Tensor swish_forward(torch::Tensor input) {
    int64_t size = input.numel();
    torch::Tensor output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    swish_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

__global__ void swish_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigma = 1.0f / (1.0f + expf(-x));
        float term = sigma + x * sigma * (1.0f - sigma);
        grad_input[idx] = grad_output[idx] * term;
    }
}

torch::Tensor swish_backward(
    torch::Tensor grad_output,
    torch::Tensor input
) {
    int64_t size = input.numel();
    torch::Tensor grad_input = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    swish_backward_kernel<<<num_blocks, block_size>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size
    );

    return grad_input;
}
"""

swish_cpp_source = """
torch::Tensor swish_forward(torch::Tensor input);
torch::Tensor swish_backward(torch::Tensor grad_output, torch::Tensor input);
"""

# Load the CUDA extension
swish_ext = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_cuda_source,
    functions=["swish_forward", "swish_backward"],
    verbose=True,
)

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return swish_ext.swish_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return swish_ext.swish_backward(grad_output, input)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SwishFunction.apply(x)

# Preserve the original get_inputs and get_init_inputs
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []