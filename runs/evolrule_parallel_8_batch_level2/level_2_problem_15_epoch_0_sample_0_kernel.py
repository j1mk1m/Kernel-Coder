import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        
        # Define fused kernel for batch norm and mean subtraction
        self.fused_kernel = load_inline(
            name="fused_kernel",
            cuda_sources=fused_kernel_source,
            functions=["fused_batch_norm_subtract"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply fused batch norm and mean subtraction
        return self.fused_kernel.fused_batch_norm_subtract(x, self.batch_norm.weight, self.batch_norm.bias, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.eps)

# CUDA kernel code for fused batch norm and mean subtraction
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_batch_norm_subtract_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> running_mean,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> running_var,
    const float eps,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output
) {
    int n = input.size(0);
    int c = input.size(1);
    int d = input.size(2);
    int h = input.size(3);
    int w = input.size(4);

    const int spatial_size = d * h * w;
    const int total_size = n * spatial_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int n_idx = idx / spatial_size;
    int s_idx = idx % spatial_size;

    int d_idx = s_idx / (h * w);
    int rem = s_idx % (h * w);
    int h_idx = rem / w;
    int w_idx = rem % w;

    for (int c_idx = 0; c_idx < c; ++c_idx) {
        scalar_t in_val = input[n_idx][c_idx][d_idx][h_idx][w_idx];
        scalar_t mean = running_mean[c_idx];
        scalar_t var = running_var[c_idx];
        scalar_t inv_std = 1.0f / sqrt(var + eps);
        
        scalar_t normed = (in_val - mean) * inv_std;
        normed = normed * weight[c_idx] + bias[c_idx];
        
        output[n_idx][c_idx][d_idx][h_idx][w_idx] = normed;
    }

    // Compute mean along spatial dimensions and subtract
    __shared__ scalar_t shared_sum[256]; // Assuming max channels <= 256

    for (int c_idx = 0; c_idx < c; c_idx += blockDim.x) {
        int tid = threadIdx.x;
        scalar_t val = (c_idx + tid < c) ? output[n_idx][c_idx + tid][d_idx][h_idx][w_idx] : 0;
        shared_sum[tid] = val;
        __syncthreads();

        // Perform reduction in shared memory
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sum[tid] += shared_sum[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            scalar_t mean = shared_sum[0] / (d * h * w);
            for (int i = 0; i < c; ++i) {
                output[n_idx][i][d_idx][h_idx][w_idx] -= mean;
            }
        }
        __syncthreads();
    }
}

torch::Tensor fused_batch_norm_subtract(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps
) {
    auto output = torch::empty_like(input);

    const int blocks = (input.numel() + 256 - 1) / 256;
    dim3 grid(blocks);
    dim3 block(256);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_batch_norm_subtract", ([&] {
        fused_batch_norm_subtract_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            weight.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            running_mean.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            running_var.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            eps,
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>()
        );
    }));

    return output;
}
"""

# Original kernel definitions remain for ConvTranspose3d and BatchNorm3d
# The fused kernel combines batch norm and mean subtraction into a single kernel
# This reduces memory traffic and kernel launch overhead

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]