import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        x[i] = val * tanh(log(exp(val) + 1));
    }
}

void mish_cuda(torch::Tensor x) {
    int n = x.numel();

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), n);
}
"""

mish_cpp_source = (
    "void mish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that uses cuDNN for convolution and custom CUDA for Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.register_buffer('conv_weight', nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)))
        self.register_buffer('conv_bias', nn.Parameter(torch.zeros(out_channels)))

    def forward(self, x):
        x = self.conv(x)
        mish_cuda(x)
        mish_cuda(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 64
    in_channels = 64
    out_channels = 128
    kernel_size = 3

    model = ModelNew(in_channels, out_channels, kernel_size)
    inputs = get_inputs()
    outputs = model(inputs[0])
    print(outputs.shape)