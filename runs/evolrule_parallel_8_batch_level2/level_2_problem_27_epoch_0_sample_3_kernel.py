import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for GroupNorm and Mean Pooling
groupnorm_mean_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_groupnorm_mean_pool_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    torch::PackedTensorAccessor<scalar_t,2> gamma,
    torch::PackedTensorAccessor<scalar_t,2> beta,
    torch::PackedTensorAccessor<scalar_t,2> output,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t num_groups,
    const float eps) {

    const int64_t group_size = channels / num_groups;
    const int64_t spatial_size = depth * height * width;

    for (int64_t b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += gridDim.x * blockDim.x) {
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t c_start = g * group_size;
            for (int64_t c = c_start; c < c_start + group_size; ++c) {
                scalar_t mean = 0.0;
                scalar_t var = 0.0;
                for (int64_t d = 0; d < depth; ++d) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            mean += input[b][c][d][h][w];
                        }
                    }
                }
                mean /= (spatial_size);
                
                for (int64_t d = 0; d < depth; ++d) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            var += (input[b][c][d][h][w] - mean) * (input[b][c][d][h][w] - mean);
                        }
                    }
                }
                var = var / spatial_size + eps;
                scalar_t std_inv = 1.0 / sqrt(var);
                
                for (int64_t d = 0; d < depth; ++d) {
                    for (int64_t h = 0; h < height; ++h) {
                        for (int64_t w = 0; w < width; ++w) {
                            scalar_t norm_val = (input[b][c][d][h][w] - mean) * std_inv;
                            input[b][c][d][h][w] = norm_val * gamma[g][c - c_start] + beta[g][c - c_start];
                        }
                    }
                }
            }
        }

        // Compute mean over spatial dimensions
        for (int64_t c = 0; c < channels; ++c) {
            scalar_t sum = 0.0;
            for (int64_t d = 0; d < depth; ++d) {
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        sum += input[b][c][d][h][w];
                    }
                }
            }
            output[b][c] = sum / spatial_size;
        }
    }
}

torch::Tensor fused_groupnorm_mean_pool_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    float eps = 1e-5) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto depth = input.size(2);
    const auto height = input.size(3);
    const auto width = input.size(4);

    auto output = torch::empty({batch_size, channels}, input.options());

    dim3 threads(256);
    dim3 blocks(1);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_groupnorm_mean_pool_cuda", ([&] {
        fused_groupnorm_mean_pool_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            gamma.packed_accessor<scalar_t,2>(),
            beta.packed_accessor<scalar_t,2>(),
            output.packed_accessor<scalar_t,2>(),
            batch_size, channels, depth, height, width, num_groups, eps);
    }));

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_groupnorm_mean_pool_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int64_t num_groups,
    float eps);
"""

# Compile the fused kernel
fused_groupnorm_mean_pool = load_inline(
    name="fused_groupnorm_mean_pool",
    cpp_sources=cpp_source,
    cuda_sources=groupnorm_mean_pool_source,
    functions=["fused_groupnorm_mean_pool_cuda"],
    verbose=True,
    extra_cflags=["-g"],
    extra_cuda_cflags=["-gencode=arch=compute_70,code=sm_70"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.gamma = self.group_norm.weight
        self.beta = self.group_norm.bias
        self.fused_groupnorm_mean_pool = fused_groupnorm_mean_pool
        self.num_groups = num_groups
        self.eps = self.group_norm.eps

    def forward(self, x):
        x = self.conv(x)
        x = F.hardswish(x)
        output = self.fused_groupnorm_mean_pool.fused_groupnorm_mean_pool_cuda(
            x, self.gamma, self.beta, self.num_groups, self.eps
        )
        return output

# Ensure get_inputs and get_init_inputs remain unchanged