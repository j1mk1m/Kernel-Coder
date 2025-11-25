import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

# Step 1: Replace GELU and AdaptiveAvgPool2d with custom CUDA kernels
# Define custom GELU kernel
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608;
    const float bias = 0.044715;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + bias * x * x * x)));
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_approx(input[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

# Define custom adaptive_avg_pool2d kernel
adaptive_avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void adaptive_avg_pool2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int output_height, int output_width) {

    const int n = input.size(0);
    const int c = input.size(1);
    const int h_in = input.size(2);
    const int w_in = input.size(3);

    const float h_step = static_cast<float>(h_in) / output_height;
    const float w_step = static_cast<float>(w_in) / output_width;

    int N = blockIdx.x * blockDim.x + threadIdx.x;
    if (N >= n) return;

    for (int C = blockIdx.y; C < c; C += gridDim.y) {
        for (int OH = 0; OH < output_height; ++OH) {
            for (int OW = 0; OW < output_width; ++OW) {
                float sum = 0.0;
                float h_start = OH * h_step;
                float h_end = (OH + 1) * h_step;
                float w_start = OW * w_step;
                float w_end = (OW + 1) * w_step;

                int h_start_idx = static_cast<int>(h_start);
                int h_end_idx = static_cast<int>(h_end);
                int w_start_idx = static_cast<int>(w_start);
                int w_end_idx = static_cast<int>(w_end);

                for (int h = h_start_idx; h <= h_end_idx; ++h) {
                    for (int w = w_start_idx; w <= w_end_idx; ++w) {
                        if (h >= 0 && h < h_in && w >= 0 && w < w_in) {
                            sum += input[N][C][h][w];
                        }
                    }
                }
                int area = (h_end_idx - h_start_idx + 1) * (w_end_idx - w_start_idx + 1);
                output[N][C][OH][OW] = sum / area;
            }
        }
    }
}

torch::Tensor adaptive_avg_pool2d_cuda(torch::Tensor input, int output_size) {
    auto output = torch::empty({input.size(0), input.size(1), output_size, output_size}, input.options());
    const int threads = 256;
    dim3 blocks(1, input.size(1));  // Adjust block and grid dimensions as needed
    adaptive_avg_pool2d_kernel<float><<<blocks, threads>>>(
        input.packed_accessor<float,4,torch::RestrictPtrTraits>(),
        output.packed_accessor<float,4,torch::RestrictPtrTraits>(),
        output_size, output_size);
    return output;
}
"""

# Compile the kernels
gelu_cuda = load_inline(
    name="gelu_cuda",
    cpp_sources="torch::Tensor gelu_cuda(torch::Tensor input);",
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
)

adaptive_avg_pool2d_cuda = load_inline(
    name="adaptive_avg_pool2d_cuda",
    cpp_sources="torch::Tensor adaptive_avg_pool2d_cuda(torch::Tensor input, int output_size);",
    cuda_sources=adaptive_avg_pool2d_source,
    functions=["adaptive_avg_pool2d_cuda"],
    verbose=True,
)

class ModelStep1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelStep1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gelu_cuda = gelu_cuda
        self.adaptive_avg_pool2d_cuda = adaptive_avg_pool2d_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu_cuda.gelu_cuda(x)
        x = self.adaptive_avg_pool2d_cuda.adaptive_avg_pool2d_cuda(x, 1)
        return x.squeeze(-1).squeeze(-1)

# Step 2: Fusing GELU and AdaptiveAvgPool2d into a single kernel
fused_gelu_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608;
    const float bias = 0.044715;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + bias * x * x * x)));
}

template <typename scalar_t>
__global__ void fused_gelu_avg_pool_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> output) {

    const int n = input.size(0);
    const int c = input.size(1);
    const int h_in = input.size(2);
    const int w_in = input.size(3);

    const float h_step = static_cast<float>(h_in);
    const float w_step = static_cast<float>(w_in);

    int N = blockIdx.x * blockDim.x + threadIdx.x;
    if (N >= n) return;

    for (int C = blockIdx.y; C < c; C += gridDim.y) {
        float sum = 0.0;
        for (int h = 0; h < h_in; ++h) {
            for (int w = 0; w < w_in; ++w) {
                float val = input[N][C][h][w];
                val = gelu_approx(val);
                sum += val;
            }
        }
        output[N][C] = sum / (h_in * w_in);
    }
}

torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input) {
    auto output = torch::empty({input.size(0), input.size(1)}, input.options());
    const int threads = 256;
    dim3 blocks(1, input.size(1));  // Adjust block and grid dimensions as needed
    fused_gelu_avg_pool_kernel<float><<<blocks, threads>>>(
        input.packed_accessor<float,4,torch::RestrictPtrTraits>(),
        output.packed_accessor<float,2,torch::RestrictPtrTraits>());
    return output;
}
"""

fused_gelu_avg_pool_cuda = load_inline(
    name="fused_gelu_avg_pool_cuda",
    cpp_sources="torch::Tensor fused_gelu_avg_pool_cuda(torch::Tensor input);",
    cuda_sources=fused_gelu_avg_pool_source,
    functions=["fused_gelu_avg_pool_cuda"],
    verbose=True,
)

class ModelStep2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelStep2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_gelu_avg_pool_cuda = fused_gelu_avg_pool_cuda

    def forward(self, x):
        x = self.conv(x)
        return self.fused_gelu_avg_pool_cuda.fused_gelu_avg_pool_cuda(x)

# Step 3: Fusing Conv output with GELU and pooling (without replacing convolution)
final_fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608;
    const float bias = 0.044715;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + bias * x * x * x)));
}

template <typename scalar_t>
__global__ void fused_gelu_avg_pool_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> output) {

    const int n = input.size(0);
    const int c = input.size(1);
    const int h_in = input.size(2);
    const int w_in = input.size(3);

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n * c; idx += blockDim.x * gridDim.x) {
        int N = idx / c;
        int C = idx % c;
        float sum = 0.0;
        for (int h = 0; h < h_in; ++h) {
            for (int w = 0; w < w_in; ++w) {
                float val = input[N][C][h][w];
                val = gelu_approx(val);
                sum += val;
            }
        }
        output[N][C] = sum / (h_in * w_in);
    }
}

torch::Tensor fused_conv_gelu_avg_pool_cuda(torch::Tensor input) {
    auto output = torch::empty({input.size(0), input.size(1)}, input.options());
    const int threads = 256;
    const int blocks = (input.size(0) * input.size(1) + threads - 1) / threads;
    fused_gelu_avg_pool_kernel<float><<<blocks, threads>>>(
        input.packed_accessor<float,4,torch::RestrictPtrTraits>(),
        output.packed_accessor<float,2,torch::RestrictPtrTraits>());
    return output;
}
"""

final_fused_kernel = load_inline(
    name="final_fused_kernel",
    cpp_sources="torch::Tensor fused_conv_gelu_avg_pool_cuda(torch::Tensor input);",
    cuda_sources=final_fused_kernel_source,
    functions=["fused_conv_gelu_avg_pool_cuda"],
    verbose=True,
)

class ModelStep3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelStep3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_conv_gelu_avg_pool_cuda = final_fused_kernel

    def forward(self, x):
        x = self.conv(x)
        return self.fused_conv_gelu_avg_pool_cuda.fused_conv_gelu_avg_pool_cuda(x)

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]