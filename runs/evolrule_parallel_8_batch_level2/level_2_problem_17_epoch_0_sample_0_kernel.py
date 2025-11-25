import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused instance norm and divide CUDA kernel
fused_instance_norm_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define EPS 1e-5  // Default epsilon for instance norm

__global__ void fused_instance_norm_divide_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float divide_by,
    float eps
) {
    // Each block handles a (n, c) group
    int n = blockIdx.x / channels;
    int c = blockIdx.x % channels;

    int spatial_size = height * width;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Compute partial sums for mean and variance
    for (int idx = tid; idx < spatial_size; idx += stride) {
        int h = idx / width;
        int w = idx % width;
        int input_offset = ((n * channels + c) * height + h) * width + w;
        float val = input[input_offset];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Use shared memory for reduction
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Reduce using parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = s_sum[0];
        float total_sum_sq = s_sum_sq[0];
        float mean = total_sum / spatial_size;
        float var = (total_sum_sq / spatial_size) - (mean * mean);
        float inv_std = 1.0f / sqrtf(var + eps);
        float scale = inv_std / divide_by;

        // Store mean and scale in shared memory for access by all threads
        s_sum[0] = mean;
        s_sum_sq[0] = scale;
    }
    __syncthreads();

    // Compute normalized values
    float mean = s_sum[0];
    float scale = s_sum_sq[0];

    for (int idx = tid; idx < spatial_size; idx += stride) {
        int h = idx / width;
        int w = idx % width;
        int output_offset = ((n * channels + c) * height + h) * width + w;
        int input_offset = output_offset; // same as input's offset
        float val = input[input_offset];
        output[output_offset] = (val - mean) * scale;
    }
}

torch::Tensor fused_instance_norm_divide_cuda(torch::Tensor input, float divide_by, float eps = EPS) {
    // Ensure input is on CUDA
    CHECK_CUDA(input);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    dim3 grid(batch_size * channels);
    dim3 block(block_size);

    fused_instance_norm_divide_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        divide_by,
        eps
    );

    return output;
}

// Utility macro to check if a tensor is CUDA
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

"""

# Define the CPP header for the function
fused_instance_norm_divide_cpp_source = """
torch::Tensor fused_instance_norm_divide_cuda(torch::Tensor input, float divide_by, float eps);
"""

# Compile the CUDA code
fused_instance_norm_divide = load_inline(
    name="fused_instance_norm_divide",
    cuda_sources=fused_instance_norm_divide_source,
    cpp_sources=fused_instance_norm_divide_cpp_source,
    functions=["fused_instance_norm_divide_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_instance_norm_divide = fused_instance_norm_divide
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_instance_norm_divide.fused_instance_norm_divide_cuda(x, self.divide_by)
        return x