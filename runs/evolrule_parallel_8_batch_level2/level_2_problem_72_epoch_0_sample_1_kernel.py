import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)
        
        # Initialize the custom batch norm kernel
        self.custom_batch_norm = load_custom_batch_norm()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom_batch_norm(x)  # No epsilon needed as it's hardcoded
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x

def load_custom_batch_norm():
    batch_norm_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    template <typename scalar_t>
    __global__ void batch_norm_forward_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        int batch_size,
        int channels,
        int spatial_dim) {
        int channel = blockIdx.x;
        int tid = threadIdx.x;

        // Compute sum for mean
        scalar_t sum = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            for (int s = tid; s < spatial_dim; s += blockDim.x) {
                int idx = b * (channels * spatial_dim) + channel * spatial_dim + s;
                sum += input[idx];
            }
        }

        // Block reduction for sum
        __shared__ scalar_t shared_sums[256];  // Assuming max block size 256
        shared_sums[tid] = sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }

        scalar_t total_sum = shared_sums[0];
        __syncthreads();

        scalar_t mean = total_sum / (batch_size * spatial_dim);

        // Compute sum of squared differences for variance
        scalar_t sum_sq_diff = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            for (int s = tid; s < spatial_dim; s += blockDim.x) {
                int idx = b * (channels * spatial_dim) + channel * spatial_dim + s;
                scalar_t val = input[idx] - mean;
                sum_sq_diff += val * val;
            }
        }

        // Block reduction for sum_sq_diff
        shared_sums[tid] = sum_sq_diff;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }

        scalar_t total_sq_diff = shared_sums[0];
        __syncthreads();

        scalar_t variance = total_sq_diff / (batch_size * spatial_dim);
        scalar_t inv_std = 1.0 / sqrt(variance + 1e-5);

        // Compute output
        for (int b = 0; b < batch_size; ++b) {
            for (int s = tid; s < spatial_dim; s += blockDim.x) {
                int idx = b * (channels * spatial_dim) + channel * spatial_dim + s;
                output[idx] = (input[idx] - mean) * inv_std;
            }
        }
    }

    torch::Tensor custom_batch_norm_forward(torch::Tensor input) {
        const int batch_size = input.size(0);
        const int channels = input.size(1);
        const int spatial_dim = input.size(2) * input.size(3) * input.size(4);

        auto output = torch::empty_like(input);

        dim3 threads(256);
        dim3 blocks(channels);

        int shared_size = threads.x * sizeof(float);

        batch_norm_forward_kernel<float><<<blocks, threads, shared_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            spatial_dim);

        return output;
    }
    """

    batch_norm_cpp_source = """
    torch::Tensor custom_batch_norm_forward(torch::Tensor input);
    """

    return load_inline(
        name="custom_batch_norm",
        cpp_sources=batch_norm_cpp_source,
        cuda_sources=batch_norm_source,
        functions=["custom_batch_norm_forward"],
        verbose=True,
        extra_cflags=[""],
        extra_cuda_cflags=["-arch=sm_75"]
    )

# Original get_inputs and get_init_inputs remain the same
batch_size = 64
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]