import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Tanh
tanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanh_backward_kernel(const float* grad_output, const float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float tanhx = tanhf(x);
        grad_input[idx] = grad_output[idx] * (1.f - tanhx * tanhx);
    }
}

torch::Tensor tanh_forward(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    tanh_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

torch::Tensor tanh_backward(torch::Tensor grad_output, torch::Tensor input) {
    auto size = grad_output.numel();
    auto grad_input = torch::empty_like(grad_output);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    tanh_backward_kernel<<<num_blocks, block_size>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), size);
    return grad_input;
}
"""

tanh_cpp_source = """
torch::Tensor tanh_forward(torch::Tensor input);
torch::Tensor tanh_backward(torch::Tensor grad_output, torch::Tensor input);
"""

# Compile the inline CUDA code for Tanh
tanh_ops = load_inline(
    name="tanh_ops",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_cuda_source,
    functions=["tanh_forward", "tanh_backward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return tanh_ops.tanh_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return tanh_ops.tanh_backward(grad_output, input)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = TanhFunction.apply(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x

def get_inputs():
    batch_size = 512
    in_channels = 64
    height = width = 32
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [64, 128, 5, 1, 1, 8, 8]