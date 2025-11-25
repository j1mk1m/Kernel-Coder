import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 4096
dim = 393216

# Define the CUDA code for SELU forward and backward
selu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// SELU constants
const float alpha = 1.6732632423543772848170429916717f;
const float lambda = 1.0507009873554804934193349852946f;

// Forward kernel
__global__ void selu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? (lambda * x) : (lambda * (alpha * expf(x) - alpha));
    }
}

torch::Tensor selu_forward_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    selu_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

// Backward kernel
__global__ void selu_backward_kernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float go = grad_output[idx];
        grad_input[idx] = (x > 0.0f) ? (lambda * go) : (lambda * alpha * expf(x) * go);
    }
}

torch::Tensor selu_backward_cuda(torch::Tensor input, torch::Tensor grad_output) {
    auto size = input.numel();
    auto grad_input = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    selu_backward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), grad_output.data_ptr<float>(), grad_input.data_ptr<float>(), size);
    return grad_input;
}
"""

selu_cuda_cpp_source = """
extern "C" {
    torch::Tensor selu_forward_cuda(torch::Tensor input);
    torch::Tensor selu_backward_cuda(torch::Tensor input, torch::Tensor grad_output);
}
"""

# Compile the CUDA code
selu_cuda = load_inline(
    name="selu_cuda",
    cpp_sources=selu_cuda_cpp_source,
    cuda_sources=selu_cuda_source,
    functions=["selu_forward_cuda", "selu_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom autograd function
class SELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return selu_cuda.selu_forward_cuda(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return selu_cuda.selu_backward_cuda(input, grad_output)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SELUFunction.apply(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []