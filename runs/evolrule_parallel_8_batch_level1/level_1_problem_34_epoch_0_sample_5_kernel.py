import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void instance_norm_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int N, int C, int H, int W,
    float eps) {

    const int n = blockIdx.x;
    const int c = blockIdx.y;

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < H * W; i += stride) {
        int h = i / W;
        int w = i % W;
        scalar_t val = input[n][c][h][w];
        local_sum += static_cast<float>(val);
        local_sum_sq += static_cast<float>(val) * static_cast<float>(val);
    }

    __syncthreads();

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Reduction phase
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float total_sum_sq = s_sum_sq[0];

    float mean = total_sum / (H * W);
    float var = total_sum_sq / (H * W) - mean * mean;
    float inv_std = 1.0f / sqrt(var + eps);

    // Compute output
    for (int i = tid; i < H * W; i += stride) {
        int h = i / W;
        int w = i % W;
        scalar_t val = input[n][c][h][w];
        output[n][c][h][w] = static_cast<scalar_t>(
            (static_cast<float>(val) - mean) * inv_std
        );
    }
}

at::Tensor instance_norm_forward_cuda(at::Tensor input, float eps) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = at::empty_like(input);

    const int block_size = 1024;
    dim3 threads(block_size);
    dim3 blocks(N, C);
    int sm_size = 2 * block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "instance_norm_forward", ([&] {
        instance_norm_forward_kernel<scalar_t><<<blocks, threads, sm_size>>>(
            input.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            N, C, H, W, eps);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

instance_norm_cpp_source = (
    "at::Tensor instance_norm_forward_cuda(at::Tensor input, float eps);"
)

# Compile the inline CUDA code
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5  # default from PyTorch's InstanceNorm2d

    def forward(self, x):
        return instance_norm.instance_norm_forward_cuda(x, self.eps)

def get_inputs():
    batch_size_val = 112
    features_val = 64
    dim1_val = 512
    dim2_val = 512
    x = torch.rand(batch_size_val, features_val, dim1_val, dim2_val).cuda()
    return [x]

def get_init_inputs():
    return [64]  # As per the original's get_inputs() dimensions