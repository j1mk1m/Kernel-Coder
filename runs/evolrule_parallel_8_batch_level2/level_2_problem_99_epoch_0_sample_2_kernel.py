import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GELU-Softmax CUDA kernel
fused_gelu_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <ATen/ATen.h>

#define APPROX_CONST 0.044715f
#define SQRT_2_OVER_PI 0.7978845608f  // Precomputed sqrt(2/pi)

__device__ float gelu(float x) {
    float poly = x * x * x * APPROX_CONST;
    float tanh_term = tanhf(SQRT_2_OVER_PI * (x + poly));
    return 0.5f * x * (1.0f + tanh_term);
}

__global__ void fused_gelu_softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int out_features
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    __shared__ float block_sum;
    __shared__ float shared_sums[256]; // Assumes blockDim.x <= 256

    float local_sum = 0.0f;

    for (int i = tid; i < out_features; i += stride) {
        float x = input[row * out_features + i];
        float y = gelu(x);
        float exp_y = expf(y);
        output[row * out_features + i] = exp_y;
        local_sum += exp_y;
    }

    __syncthreads();

    shared_sums[tid] = local_sum;
    __syncthreads();

    // Reduction step
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum = shared_sums[0];
    }
    __syncthreads();

    for (int i = tid; i < out_features; i += stride) {
        output[row * out_features + i] /= block_sum;
    }
}

torch::Tensor fused_gelu_softmax_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int out_features = input.size(1);

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const dim3 blocks(batch_size);
    const dim3 threads(threads_per_block);

    fused_gelu_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    return output;
}
"""

fused_gelu_softmax_cpp_source = (
    "torch::Tensor fused_gelu_softmax_cuda(torch::Tensor input);"
)

# Compile the fused kernel
fused_gelu_softmax = load_inline(
    name="fused_gelu_softmax",
    cpp_sources=fused_gelu_softmax_cpp_source,
    cuda_sources=fused_gelu_softmax_source,
    functions=["fused_gelu_softmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_gelu_softmax = fused_gelu_softmax

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_gelu_softmax.fused_gelu_softmax_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features]