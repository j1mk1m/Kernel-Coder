import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_features
        self.eps = 1e-5  # Assuming default epsilon value

        # Define the CUDA kernel
        kernel_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

template <typename scalar_t>
__global__ void group_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_groups,
    const int channels_per_group,
    const int spatial_size,
    const float eps) {

    const int group_id = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sum_sq = sdata + blockDim.x;

    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    __syncthreads();

    const int elements_per_group = batch_size * channels_per_group * spatial_size;
    const int elements_per_thread = (elements_per_group + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < elements_per_thread; ++i) {
        const int pos = tid + i * blockDim.x;
        if (pos < elements_per_group) {
            const int c = (pos / (batch_size * spatial_size)) % channels_per_group;
            const int b = (pos / (channels_per_group * spatial_size)) % batch_size;
            const int s = pos % spatial_size;

            const int offset = b * num_groups * channels_per_group * spatial_size
                             + group_id * channels_per_group * spatial_size
                             + c * spatial_size
                             + s;

            scalar_t val = input[offset];
            s_sum[tid] += static_cast<float>(val);
            s_sum_sq[tid] += static_cast<float>(val * val);
        }
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float total = static_cast<float>(elements_per_group);
        float mean = s_sum[0] / total;
        float var = s_sum_sq[0] / total - mean * mean;
        var = fmaxf(var, eps);
        float inv_std = rsqrtf(var + eps);

        s_sum[0] = mean;
        s_sum_sq[0] = inv_std;
    }
    __syncthreads();

    for (int i = 0; i < elements_per_thread; ++i) {
        const int pos = tid + i * blockDim.x;
        if (pos < elements_per_group) {
            const int c = (pos / (batch_size * spatial_size)) % channels_per_group;
            const int b = (pos / (channels_per_group * spatial_size)) % batch_size;
            const int s = pos % spatial_size;

            const int offset = b * num_groups * channels_per_group * spatial_size
                             + group_id * channels_per_group * spatial_size
                             + c * spatial_size
                             + s;

            output[offset] = (input[offset] - s_sum[0]) * s_sum_sq[0];
        }
    }
}

torch::Tensor group_norm_cuda(torch::Tensor input, int num_groups, int num_channels, float eps) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto spatial_size = input.numel() / (batch_size * channels);
    const auto channels_per_group = channels / num_groups;

    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const dim3 blocks(num_groups);
    const int shared_size = 2 * threads_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "group_norm_cuda", ([&] {
        group_norm_kernel<scalar_t><<<blocks, threads_per_block, shared_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_groups,
            channels_per_group,
            spatial_size,
            eps);
    }));

    return output;
}
"""

        # Load the CUDA kernel
        self.group_norm = load_inline(
            name="group_norm",
            cpp_sources=f"""
                #include <torch/extension.h>
                #include <cuda.h>
                #include <cuda_runtime.h>
            """,
            cuda_sources=kernel_code,
            functions=["group_norm_cuda"],
            verbose=True,
            with_cuda=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm.group_norm_cuda(
            x,
            self.num_groups,
            self.num_channels,
            self.eps
        )

# The rest of the code (get_inputs, get_init_inputs) remains unchanged as per original