import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom GEMM (Linear layer) kernel
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void custom_gemm_kernel(const scalar_t* __restrict__ input,
                                  const scalar_t* __restrict__ weight,
                                  scalar_t* __restrict__ output,
                                  int batch_size,
                                  int in_features,
                                  int out_features) {
    int batch = blockIdx.x;
    int out_idx = threadIdx.x;

    if (out_idx >= out_features) return;

    scalar_t sum = 0;
    for (int i = 0; i < in_features; i++) {
        sum += input[batch * in_features + i] * weight[out_idx * in_features + i];
    }

    output[batch * out_features + out_idx] = sum;
}

torch::Tensor custom_gemm_cuda(torch::Tensor input, torch::Tensor weight) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int threads = 256;
    const dim3 blocks(batch_size);
    const dim3 threads_per_block(std::min(out_features, threads));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_gemm_cuda", ([&] {
        custom_gemm_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Custom Swish kernel (x * sigmoid(x))
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void custom_swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_val = 1.0 / (1.0 + expf(-x[idx]));
        out[idx] = x[idx] * sigmoid_val;
    }
}

torch::Tensor custom_swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_swish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

# Custom Divide kernel (x / 2.0)
divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_divide_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / 2.0f;
    }
}

torch::Tensor custom_divide_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_divide_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

# Custom Clamp kernel (min=-1, max=1)
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_clamp_kernel(const float* x, float* out, int size, float min, float max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(fminf(x[idx], max), min);
    }
}

torch::Tensor custom_clamp_cuda(torch::Tensor x, float min, float max) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_clamp_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size, min, max);

    return out;
}
"""

# Custom Tanh kernel
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_tanh_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(x[idx]);
    }
}

torch::Tensor custom_tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_tanh_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

# Compile all kernels
gemm_cpp = "torch::Tensor custom_gemm_cuda(torch::Tensor input, torch::Tensor weight);"
gemm = load_inline(
    name="custom_gemm",
    cpp_sources=gemm_cpp,
    cuda_sources=gemm_source,
    functions=["custom_gemm_cuda"],
    verbose=True
)

swish_cpp = "torch::Tensor custom_swish_cuda(torch::Tensor x);"
swish = load_inline(
    name="custom_swish",
    cpp_sources=swish_cpp,
    cuda_sources=swish_source,
    functions=["custom_swish_cuda"],
    verbose=True
)

divide_cpp = "torch::Tensor custom_divide_cuda(torch::Tensor x);"
divide = load_inline(
    name="custom_divide",
    cpp_sources=divide_cpp,
    cuda_sources=divide_source,
    functions=["custom_divide_cuda"],
    verbose=True
)

clamp_cpp = "torch::Tensor custom_clamp_cuda(torch::Tensor x, float min, float max);"
clamp = load_inline(
    name="custom_clamp",
    cpp_sources=clamp_cpp,
    cuda_sources=clamp_source,
    functions=["custom_clamp_cuda"],
    verbose=True
)

tanh_cpp = "torch::Tensor custom_tanh_cuda(torch::Tensor x);"
tanh = load_inline(
    name="custom_tanh",
    cpp_sources=tanh_cpp,
    cuda_sources=tanh_source,
    functions=["custom_tanh_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        self.gemm = gemm
        self.swish = swish
        self.divide = divide
        self.clamp = clamp
        self.tanh = tanh

    def forward(self, x):
        # Custom GEMM (Linear layer) with bias
        x = self.gemm.custom_gemm_cuda(x, self.weight)
        if self.bias is not None:
            x += self.bias.unsqueeze(0).expand(x.size(0), -1)
        
        # Swish activation
        x = self.swish.custom_swish_cuda(x)
        
        # Divide by 2.0
        x = self.divide.custom_divide_cuda(x)
        
        # First Clamp
        x = self.clamp.custom_clamp_cuda(x, -1.0, 1.0)
        
        # Tanh activation
        x = self.tanh.custom_tanh_cuda(x)
        
        # Second Clamp
        x = self.clamp.custom_clamp_cuda(x, -1.0, 1.0)
        return x

# Ensure the input and initialization functions are unchanged
def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]

# Define the constants as in the original code
batch_size = 1024
in_features = 8192
out_features = 8192