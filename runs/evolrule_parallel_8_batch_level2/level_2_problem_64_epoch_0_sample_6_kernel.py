import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused post-processing kernel
fused_post_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename T>
struct MaxOp {
    __host__ __device__ __forceinline__
    T operator()(const T a, const T b) const {
        return fmaxf(a, b);
    }
};

template <typename T>
struct SumExpOp {
    __host__ __device__ __forceinline__
    T operator()(const T a, const T b) const {
        return a + b;
    }
};

template <typename T>
__device__ T leaky_relu(T x, T neg_slope) {
    return x > 0 ? x : neg_slope * x;
}

template <typename T>
__device__ T gelu_approx(T x) {
    const T sqrt_2_over_pi = 0.7978845608;
    const T a = 0.044715;
    T inner = sqrt_2_over_pi * (x + a * x * x * x);
    return 0.5f * x * (1 + tanhf(inner));
}

extern "C" __global__ void fused_post_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_features
) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float val = input[sample_idx * out_features + j];
        if (val > max_val) {
            max_val = val;
        }
    }

    float block_max = BlockReduce(temp_storage).Reduce(max_val, MaxOp<float>());
    if (threadIdx.x == 0) max_val = block_max;
    __syncthreads();

    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        float temp = input[sample_idx * out_features + j] - max_val;
        sum_exp += exp(temp);
    }

    float block_sum_exp = BlockReduce(temp_storage).Reduce(sum_exp, SumExpOp<float>());
    if (threadIdx.x == 0) {
        float lse = max_val + logf(block_sum_exp);
        lse = leaky_relu(leaky_relu(lse, 0.01f), 0.01f);
        lse = gelu_approx(gelu_approx(lse));
        output[sample_idx] = lse;
    }
}

torch::Tensor fused_post_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int out_features = input.size(1);

    auto output = torch::empty({batch_size, 1}, input.options());

    dim3 blocks(batch_size);
    dim3 threads(256);  // Block size must be <= 1024 and divide out_features evenly?

    // Launch kernel with block size 256, but need to ensure out_features is divisible by threads.x?
    // Alternatively, use dynamic loop
    fused_post_kernel<<<blocks, threads>>>(
        input.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features
    );

    cudaDeviceSynchronize();
    return output;
}
"""

fused_post_cpp_source = "torch::Tensor fused_post_cuda(torch::Tensor input);"

# Compile the fused post-processing kernel
fused_post_op = load_inline(
    name="fused_post",
    cpp_sources=fused_post_cpp_source,
    cuda_sources=fused_post_source,
    functions=["fused_post_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_post = fused_post_op

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_post.fused_post_cuda(x)
        return x.view(-1, 1)  # Ensure output shape is [batch, 1]

def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]