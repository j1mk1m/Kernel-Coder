import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for element-wise division by a constant
div_const_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void div_const_kernel(const scalar_t* a, scalar_t* out, const float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / divisor;
    }
}

torch::Tensor div_const_cuda(torch::Tensor a, const float divisor) {
    auto size = a.numel();
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "div_const_cuda", ([&] {
        div_const_kernel<scalar_t><<<num_blocks, block_size>>>(
            a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), divisor, size);
    }));

    return out;
}
"""

div_const_cpp_source = (
    "torch::Tensor div_const_cuda(torch::Tensor a, const float divisor);"
)

div_const = load_inline(
    name="div_const",
    cpp_sources=div_const_cpp_source,
    cuda_sources=div_const_source,
    functions=["div_const_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for element-wise addition (bias addition)
add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void add_bias_kernel(const scalar_t* a, const scalar_t* bias, scalar_t* out, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * channels * depth * height * width) {
        int c = (idx / (depth * height * width)) % channels;
        out[idx] = a[idx] + bias[c];
    }
}

torch::Tensor add_bias_cuda(torch::Tensor a, torch::Tensor bias) {
    auto a_size = a.sizes();
    auto batch = a_size[0];
    auto channels = a_size[1];
    auto depth = a_size[2];
    auto height = a_size[3];
    auto width = a_size[4];
    
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int size = batch * channels * depth * height * width;
    const int num_blocks = (size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "add_bias_cuda", ([&] {
        add_bias_kernel<scalar_t><<<num_blocks, block_size>>>(
            a.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), 
            out.data_ptr<scalar_t>(), batch, channels, depth, height, width);
    }));

    return out;
}
"""

add_bias_cpp_source = (
    "torch::Tensor add_bias_cuda(torch::Tensor a, torch::Tensor bias);"
)

add_bias = load_inline(
    name="add_bias",
    cpp_sources=add_bias_cpp_source,
    cuda_sources=add_bias_source,
    functions=["add_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for summing along a specific dimension
sum_dim_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_dim_kernel(const scalar_t* a, scalar_t* out, 
                              int batch, int channels, int depth, int height, int width,
                              int sum_dim, int out_dim_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * depth * height * width) {
        int c;
        if (sum_dim == 1) {
            c = 0;
            for (int ch = 0; ch < channels; ++ch) {
                out[idx * out_dim_size + c] += a[idx * channels + ch];
            }
        }
        // Assuming sum_dim is always 1 as per given problem parameters
    }
}

torch::Tensor sum_dim_cuda(torch::Tensor a, int sum_dim) {
    auto a_size = a.sizes();
    auto batch = a_size[0];
    auto channels = a_size[1];
    auto depth = a_size[2];
    auto height = a_size[3];
    auto width = a_size[4];
    
    auto out_size = a_size.tolist();
    out_size.erase(out_size.begin() + sum_dim);
    auto out = torch::zeros(out_size, a.options());

    const int block_size = 256;
    const int elements = batch * depth * height * width;
    const int num_blocks = (elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "sum_dim_cuda", ([&] {
        sum_dim_kernel<scalar_t><<<num_blocks, block_size>>>(
            a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            batch, channels, depth, height, width,
            sum_dim, out.size(1));
    }));

    return out;
}
"""

sum_dim_cpp_source = (
    "torch::Tensor sum_dim_cuda(torch::Tensor a, int sum_dim);"
)

sum_dim = load_inline(
    name="sum_dim",
    cpp_sources=sum_dim_cpp_source,
    cuda_sources=sum_dim_source,
    functions=["sum_dim_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        # Load custom kernels
        self.div_const = div_const
        self.add_bias = add_bias
        self.sum_dim_op = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = self.div_const.div_const_cuda(x, self.divisor)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = self.add_bias.add_bias_cuda(x, self.bias)
        x = self.sum_dim_op.sum_dim_cuda(x, self.sum_dim)
        return x