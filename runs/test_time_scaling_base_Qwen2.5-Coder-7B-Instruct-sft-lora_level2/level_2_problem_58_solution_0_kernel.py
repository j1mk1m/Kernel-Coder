import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for logsumexp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsumexp_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float max_val = x[idx];
        for (int i = 1; i < size; ++i) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_exp += exp(x[i] - max_val);
        }
        y[idx] = max_val + log(sum_exp);
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

logsumexp_cpp_source = (
    "torch::Tensor logsumexp_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for logsumexp
logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for hardswish
hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x[idx] * (x[idx] > 0 && x[idx] < 6 ? 0.5 : 1);
    }
}

torch::Tensor hardswish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

hardswish_cpp_source = (
    "torch::Tensor hardswish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for hardswish
hardswish = load_inline(
    name="hardswish",
    cpp_sources=hardswish_cpp_source,
    cuda_sources=hardswish_source,
    functions=["hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for clamp
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(const float* x, float* y, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = fmaxf(fminf(x[idx], max_val), min_val);
    }
}

torch::Tensor clamp_cuda(torch::Tensor x, float min_val, float max_val) {
    auto size = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size, min_val, max_val);

    return y;
}
"""

clamp_cpp_source = (
    "torch::Tensor clamp_cuda(torch::Tensor x, float min_val, float max_val);"
)

# Compile the inline CUDA code for clamp
clamp = load_inline(
    name="clamp",
    cpp_sources=clamp_cpp_source,
    cuda_sources=clamp_source,
    functions=["clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for logsumexp, hardswish, and clamp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = logsumexp.logsumexp_cuda(x)
        x = hardswish.hardswish_cuda(x)
        x = x - self.bias
        x = clamp.clamp_cuda(x, -1, 1)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 3
    out_channels = 16
    depth, height, width = 16, 32, 32
    kernel_size = 3
    stride = 2
    padding = 1
    bias_shape = (1, 1, 1, 1)

    model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, bias_shape)
    inputs = get_inputs()
    output = model_new(inputs[0])
    print(output.shape)