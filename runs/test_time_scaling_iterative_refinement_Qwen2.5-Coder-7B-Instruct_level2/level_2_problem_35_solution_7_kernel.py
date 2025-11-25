import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise subtraction
elementwise_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_subtract_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor elementwise_subtract_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_subtract_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_subtract_cpp_source = (
    "torch::Tensor elementwise_subtract_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise subtraction
elementwise_subtract = load_inline(
    name="elementwise_subtract",
    cpp_sources=elementwise_subtract_cpp_source,
    cuda_sources=elementwise_subtract_source,
    functions=["elementwise_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for fused hardswish and mish
fused_hswish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_hswish_mish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float hswish_val = val * std::min(std::max(val + 3.0f, 0.0f), 6.0f) / 6.0f;
        float mish_val = val * tanh(log(1.0f + exp(hswish_val)));
        out[idx] = mish_val;
    }
}

torch::Tensor fused_hswish_mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_hswish_mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_hswish_mish_cpp_source = (
    "torch::Tensor fused_hswish_mish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for fused hardswish and mish
fused_hswish_mish = load_inline(
    name="fused_hswish_mish",
    cpp_sources=fused_hswish_mish_cpp_source,
    cuda_sources=fused_hswish_mish_source,
    functions=["fused_hswish_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for max pooling
max_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pooling_kernel(const float* x, float* out, int batch_size, int channels, int height, int width, int pool_height, int pool_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;

    int b = idx / (channels * height * width);
    int c = (idx % (channels * height * width)) / (height * width);
    int h = (idx % (channels * height * width)) % height;
    int w = (idx % (channels * height * width)) % width;

    int ph_start = h / pool_height;
    int pw_start = w / pool_width;
    int ph_end = std::min(ph_start + pool_height, height);
    int pw_end = std::min(pw_start + pool_width, width);

    float max_val = -std::numeric_limits<float>::infinity();
    for (int ph = ph_start; ph < ph_end; ++ph) {
        for (int pw = pw_start; pw < pw_end; ++pw) {
            int i = b * channels * height * width + c * height * width + ph * width + pw;
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
    }

    int oh = h / pool_height;
    int ow = w / pool_width;
    int o_idx = b * channels * (height / pool_height) * (width / pool_width) + c * (height / pool_height) * (width / pool_width) + oh * (width / pool_width) + ow;
    out[o_idx] = max_val;
}

torch::Tensor max_pooling_cuda(torch::Tensor x, int pool_height, int pool_width) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out = torch::zeros({batch_size, channels, height / pool_height, width / pool_width}, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height * width + block_size - 1) / block_size;

    max_pooling_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, channels, height, width, pool_height, pool_width);

    return out;
}
"""

max_pooling_cpp_source = (
    "torch::Tensor max_pooling_cuda(torch::Tensor x, int pool_height, int pool_width);"
)

# Compile the inline CUDA code for max pooling
max_pooling = load_inline(
    name="max_pooling",
    cpp_sources=max_pooling_cpp_source,
    cuda_sources=max_pooling_source,
    functions=["max_pooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = elementwise_subtract.elementwise_subtract_cuda(x, torch.full_like(x, self.subtract_value))
        x = fused_hswish_mish.fused_hswish_mish_cuda(x)
        x = max_pooling.max_pooling_cuda(x, self.pool.kernel_size[0], self.pool.kernel_size[1])
        return x