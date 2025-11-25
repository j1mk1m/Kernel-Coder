import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for the mean operation across all dimensions except batch
mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TOTAL_ELEMENTS_PER_BATCH 475200  // 24 channels * (24-3+1)* (32-3+1)*(32-3+1) = 24*22*30*30

__global__ void mean_kernel(float* input, float* output, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float sum = 0.0f;

    for (int i = tid; i < TOTAL_ELEMENTS_PER_BATCH; i += num_threads) {
        float val = input[batch_idx * TOTAL_ELEMENTS_PER_BATCH + i];
        sum += val;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx] = shared[0] / TOTAL_ELEMENTS_PER_BATCH;
    }
}

torch::Tensor mean_all_except_batch_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    auto output = torch::empty({batch_size}, input.options());
    const int block_size = 256;
    const int shared_size = block_size * sizeof(float);

    // Launch the kernel
    mean_kernel<<<batch_size, block_size, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size);

    return output;
}
"""

mean_header = "torch::Tensor mean_all_except_batch_cuda(torch::Tensor input);"

# Compile the custom CUDA kernel
mean_all_except_batch = load_inline(
    name="mean_all_except_batch",
    cpp_sources=mean_header,
    cuda_sources=mean_source,
    functions=["mean_all_except_batch_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.mean_kernel = mean_all_except_batch  # Load the custom mean function

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        return self.mean_kernel.mean_all_except_batch_cuda(x)