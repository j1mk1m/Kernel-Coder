import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

class FusedHardSwishGroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, num_groups):
        # Define CUDA kernel for fused HardSwish and GroupNorm
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void fused_hardswish_groupnorm_forward(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            scalar_t* __restrict__ output,
            int batch_size,
            int channels,
            int spatial_size,
            int num_groups) {
            extern __shared__ scalar_t shared[];
            int batch_idx = blockIdx.x;
            int group_idx = threadIdx.x / (channels / num_groups);
            int channel_in_group = threadIdx.x % (channels / num_groups);
            int spatial_idx = blockIdx.y;

            // Compute HardSwish activation
            scalar_t x = input[batch_idx * channels * spatial_size + 
                              (group_idx * (channels / num_groups) + channel_in_group) * spatial_size + 
                              spatial_idx];
            scalar_t hswish = x * (x + 3.f) / (6.f + 6.f * fmaxf(fminf(x + 3.f, 6.f), 0.f));

            // Accumulate for mean and variance (per group and spatial dimensions)
            int offset = threadIdx.x * spatial_size;
            shared[offset + spatial_idx] = hswish;
            __syncthreads();

            scalar_t sum = 0, sum_sq = 0;
            for (int s = 0; s < spatial_size; ++s) {
                scalar_t val = shared[offset + s];
                sum += val;
                sum_sq += val * val;
            }
            __syncthreads();

            // Reduction across spatial dimensions (could use warp-level reduction for better performance)
            scalar_t mean = sum / spatial_size;
            scalar_t var = sum_sq / spatial_size - mean * mean;
            scalar_t std = rsqrtf(var + 1e-5f);

            // Normalize and apply scale/shift
            scalar_t w = weight[group_idx * (channels / num_groups) + channel_in_group];
            scalar_t b = bias[group_idx * (channels / num_groups) + channel_in_group];
            scalar_t normalized = (hswish - mean) * std * w + b;

            // Write to output (assuming output is stored as [B, C, D, H, W])
            output[batch_idx * channels * spatial_size + 
                  (group_idx * (channels / num_groups) + channel_in_group) * spatial_size + 
                  spatial_idx] = normalized;
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_hardswish_groupnorm_forward_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int num_groups) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int depth = input.size(2);
            const int height = input.size(3);
            const int width = input.size(4);
            const int spatial_size = depth * height * width;
            const int groups = num_groups;
            const int threads = channels; // Each channel in group is handled by a thread
            const int blocks = batch_size * spatial_size;

            auto output = torch::empty_like(input);
            auto mean = torch::empty(batch_size, channels, 1).to(input.device());
            auto var = torch::empty(batch_size, channels, 1).to(input.device());

            // Launch kernel
            dim3 grid(blocks);
            dim3 block(threads);
            const int shared_mem = threads * spatial_size * sizeof(float);
            AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_hardswish_groupnorm_forward", ([&] {
                fused_hardswish_groupnorm_forward<scalar_t><<<grid, block, shared_mem>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    spatial_size,
                    num_groups);
            }));

            return std::make_tuple(output, mean, var);
        }

        // Backward pass would be needed, but for simplicity omitted here
        // This is a simplified forward-only example for demonstration purposes
        """

        # Compile the fused kernel
        fused_kernel = load_inline(
            name="fused_hardswish_groupnorm",
            cpp_sources="",
            cuda_sources=fused_kernel_source,
            functions=["fused_hardswish_groupnorm_forward_cuda"],
            verbose=True,
        )

        # Execute the kernel
        output, mean, var = fused_kernel.fused_hardswish_groupnorm_forward_cuda(
            input, weight, bias, num_groups
        )
        ctx.save_for_backward(input, weight, bias, mean, var)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass here (simplified for brevity)
        # This is a placeholder, actual implementation requires detailed calculation
        input, weight, bias, mean, var = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        return grad_input, grad_weight, grad_bias, None

class FusedMeanPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Define CUDA kernel for mean pooling over spatial dimensions
        mean_pool_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void mean_pool_forward(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            int batch_size,
            int channels,
            int depth,
            int height,
            int width) {
            int batch_idx = blockIdx.x;
            int channel_idx = threadIdx.x;

            scalar_t sum = 0;
            for (int d = 0; d < depth; ++d) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        sum += input[batch_idx * channels * depth * height * width +
                                    channel_idx * depth * height * width +
                                    d * height * width + h * width + w];
                    }
                }
            }
            output[batch_idx * channels + channel_idx] = sum / (depth * height * width);
        }

        torch::Tensor mean_pool_forward_cuda(torch::Tensor input) {
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int depth = input.size(2);
            const int height = input.size(3);
            const int width = input.size(4);
            const int spatial_size = depth * height * width;

            auto output = torch::empty({batch_size, channels}, device=input.device());
            dim3 grid(batch_size);
            dim3 block(channels);
            AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_pool_forward", ([&] {
                mean_pool_forward<scalar_t><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    depth,
                    height,
                    width);
            }));

            return output;
        }
        """

        # Compile the fused mean pooling kernel
        mean_pool = load_inline(
            name="mean_pool",
            cuda_sources=mean_pool_source,
            functions=["mean_pool_forward_cuda"],
            verbose=True,
        )

        output = mean_pool.mean_pool_forward_cuda(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass here (simplified)
        input, = ctx.saved_tensors
        batch_size, channels, depth, height, width = input.size()
        grad_input = grad_output.view(batch_size, channels, 1, 1, 1).expand_as(input) / (depth * height * width)
        return grad_input

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)  # Keep parameters
        self.fused_hswish_gn = FusedHardSwishGroupNorm.apply
        self.fused_mean = FusedMeanPool.apply

    def forward(self, x):
        x = self.conv(x)
        # Fuse HardSwish and GroupNorm
        x = self.fused_hswish_gn(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups
        )
        # Fuse mean pooling
        x = self.fused_mean(x)
        return x

# Ensure that parameters are properly accessible
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]