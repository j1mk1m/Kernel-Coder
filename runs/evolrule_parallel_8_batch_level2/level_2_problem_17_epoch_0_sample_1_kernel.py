import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instance_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void instance_norm_div_kernel(
    const float* input, 
    float* output, 
    float divide_by,
    int batch_size, 
    int channels, 
    int height, 
    int width) {

    int sample = blockIdx.x;
    int channel = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    float* s_sum = shared;
    float* s_sum_sq = s_sum + blockDim.x;

    // Initialize shared memory
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    __syncthreads();

    // Compute partial sums for each thread's spatial elements
    for (int h = tid; h < height * width; h += blockDim.x) {
        int w = h % width;
        int h_idx = h / width;
        int idx = sample * channels * height * width +
                  channel * height * width +
                  h_idx * width + w;
        float val = input[idx];
        s_sum[tid] += val;
        s_sum_sq[tid] += val * val;
    }
    __syncthreads();

    // Reduce within the block to compute total sum and sum_sq
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float total_sum_sq = s_sum_sq[0];
    __syncthreads();

    if (tid == 0) {
        float count = height * width;
        float mean = total_sum / count;
        float variance = (total_sum_sq / count) - mean * mean;
        float inv_std = 1.0f / sqrt(variance + 1e-5f);
        shared[0] = mean;
        shared[1] = inv_std;
    }
    __syncthreads();

    // Now, all threads can compute the normalized value
    for (int h = tid; h < height * width; h += blockDim.x) {
        int w = h % width;
        int h_idx = h / width;
        int idx = sample * channels * height * width +
                  channel * height * width +
                  h_idx * width + w;
        float val = input[idx];
        output[idx] = (val - shared[0]) * shared[1] / divide_by;
    }
}

torch::Tensor instance_norm_div_cuda(torch::Tensor input, float divide_by) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    dim3 blocks(batch_size, channels);
    dim3 threads(block_size);

    int shared_size = (block_size + 2) * sizeof(float);
    instance_norm_div_kernel<<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        divide_by,
        batch_size,
        channels,
        height,
        width);

    return output;
}
"""

instance_norm_div_cpp_source = (
    "torch::Tensor instance_norm_div_cuda(torch::Tensor input, float divide_by);"
)

instance_norm_div = load_inline(
    name="instance_norm_div",
    cpp_sources=instance_norm_div_cpp_source,
    cuda_sources=instance_norm_div_source,
    functions=["instance_norm_div_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.instance_norm_div = instance_norm_div

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm_div.instance_norm_div_cuda(x, self.divide_by)
        return x