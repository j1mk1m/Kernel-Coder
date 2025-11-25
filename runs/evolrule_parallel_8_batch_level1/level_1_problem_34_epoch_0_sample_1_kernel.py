import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

epsilon = 1e-5

# Custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__device__ inline scalar_t warp_reduce(scalar_t val) {
    for (int mask = 1; mask != 0x80; mask <<= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, mask);
    }
    return val;
}

template <typename scalar_t>
__global__ void instance_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y,
    const int N, const int C, const int H, const int W,
    const float eps) {

    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int feature_idx = n * C + c;

    // Compute mean and variance for each (n, c) slice
    const int spatial_size = H * W;
    const int tid = threadIdx.x;

    extern __shared__ scalar_t shared_data[];

    scalar_t* buffer = shared_data;
    scalar_t* sum_buf = buffer;
    scalar_t* sq_sum_buf = sum_buf + blockDim.x;

    scalar_t sum = 0.0;
    scalar_t sq_sum = 0.0;

    for (int i = tid; i < spatial_size; i += blockDim.x) {
        const int idx = n * C * H * W + c * H * W + i;
        scalar_t val = x[idx];
        sum += val;
        sq_sum += val * val;
    }

    sum = warp_reduce<scalar_t>(sum);
    sq_sum = warp_reduce<scalar_t>(sq_sum);

    if (tid == 0) {
        sum_buf[blockIdx.z] = sum;
        sq_sum_buf[blockIdx.z] = sq_sum;
    }
    __syncthreads();

    if (tid < 32) {
        sum = 0.0;
        sq_sum = 0.0;
        for (int i = tid; i < blockDim.x; i += 32) {
            sum += sum_buf[i];
            sq_sum += sq_sum_buf[i];
        }
        sum = warp_reduce<scalar_t>(sum);
        sq_sum = warp_reduce<scalar_t>(sq_sum);
        if (tid == 0) {
            sum_buf[0] = sum;
            sq_sum_buf[0] = sq_sum;
        }
    }
    __syncthreads();

    sum = sum_buf[0];
    sq_sum = sq_sum_buf[0];

    const scalar_t mean = sum / spatial_size;
    const scalar_t var = sq_sum / spatial_size - mean * mean;
    const scalar_t rstd = 1.0 / sqrt(var + eps);

    // Normalize and scale
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        const int idx = n * C * H * W + c * H * W + i;
        scalar_t val = x[idx];
        y[idx] = (val - mean) * rstd * weight[c] + bias[c];
    }
}

at::Tensor instance_norm_forward_cuda(
    const at::Tensor x,
    const at::Tensor weight,
    const at::Tensor bias) {

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto y = at::empty_like(x);
    const int threads = 256;
    dim3 blocks(N, C, 1);
    dim3 threads_per_block(threads, 1, 1);

    const int shared_size = 2 * threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "instance_norm_forward_cuda", ([&] {
        instance_norm_forward_kernel<scalar_t><<<blocks, threads_per_block, shared_size>>>(
            x.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias.data<scalar_t>(),
            y.data<scalar_t>(),
            N, C, H, W,
            epsilon);
    }));

    cudaDeviceSynchronize();
    return y;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias);
"""

instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.instance_norm = instance_norm

    def forward(self, x):
        return self.instance_norm.instance_norm_forward_cuda(
            x, self.weight, self.bias
        )

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x.cuda()]

def get_init_inputs():
    return [features]