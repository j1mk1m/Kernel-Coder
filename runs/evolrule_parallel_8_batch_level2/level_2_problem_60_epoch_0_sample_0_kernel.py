import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the Swish CUDA kernel
swish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void swish_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

torch::Tensor swish_forward_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    return output;
}
"""

swish_cpp = "torch::Tensor swish_forward_cuda(torch::Tensor input);"

swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp,
    cuda_sources=swish_source,
    functions=["swish_forward_cuda"],
    verbose=True,
)

# Define the HardSwish CUDA kernel
hardswish_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void hardswish_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float tmp = x + 3.0f;
        tmp = tmp > 6.0f ? 6.0f : (tmp < 0.0f ? 0.0f : tmp);
        output[idx] = x * tmp / 6.0f;
    }
}

torch::Tensor hardswish_forward_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    return output;
}
"""

hardswish_cpp = "torch::Tensor hardswish_forward_cuda(torch::Tensor input);"

hardswish = load_inline(
    name="hardswish",
    cpp_sources=hardswish_cpp,
    cuda_sources=hardswish_source,
    functions=["hardswish_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps
        )
        # The Swish and HardSwish are replaced by the custom CUDA functions
        self.swish = swish
        self.hardswish = hardswish

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.swish.swish_forward_cuda(x)  # Custom Swish
        x = self.group_norm(x)
        x = self.hardswish.hardswish_forward_cuda(x)  # Custom HardSwish
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]