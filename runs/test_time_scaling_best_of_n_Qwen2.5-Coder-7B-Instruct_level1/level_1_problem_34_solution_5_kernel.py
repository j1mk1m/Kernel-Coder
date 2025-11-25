import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for instance normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* input, float* mean, float* var, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) {
        return;
    }

    int c_idx = idx % channels;
    int h_idx = (idx / channels) % height;
    int w_idx = (idx / (channels * height)) % width;
    int b_idx = idx / (channels * height * width);

    float val = input[idx];

    // Compute mean and variance
    atomicAdd(&mean[c_idx], val);
    atomicAdd(&var[c_idx], val * val);
}

void compute_mean_var(float* mean, float* var, int batch_size, int channels) {
    for (int i = 0; i < channels; ++i) {
        mean[i] /= batch_size * height * width;
        var[i] /= batch_size * height * width;
        var[i] -= mean[i] * mean[i];
        var[i] = max(var[i], 1e-5f);
    }
}

__global__ void apply_instance_norm_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) {
        return;
    }

    int c_idx = idx % channels;
    int h_idx = (idx / channels) % height;
    int w_idx = (idx / (channels * height)) % width;
    int b_idx = idx / (channels * height * width);

    float val = input[idx];
    float normed_val = (val - mean[c_idx]) / sqrt(var[c_idx]);
    output[idx] = normed_val;
}
"""

instance_norm_cpp_source = (
    "void instance_norm_cuda(torch::Tensor input, torch::Tensor output);"
)

# Compile the inline CUDA code for instance normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        mean = torch.zeros(channels).cuda()
        var = torch.zeros(channels).cuda()

        self.instance_norm.instance_norm_cuda(x.contiguous(), mean, var, output)

        # Apply normalization using computed mean and variance
        for c in range(channels):
            output[:, c, :, :] = (output[:, c, :, :] - mean[c]) / torch.sqrt(var[c])

        return output