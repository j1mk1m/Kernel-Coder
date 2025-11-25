import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused InstanceNorm + division kernel
fused_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDACachingAllocator.h>

// Define the CUDA kernel for fusing instance norm and division
template <typename T>
__global__ void fused_instance_norm_div(
    const T* input, T* output,
    const T* weight, const T* bias,
    const T* running_mean, const T* running_var,
    int batch, int channels, int spatial_dim,
    T eps, T divide_by) {

    extern __shared__ char sdata[];
    T* shared = reinterpret_cast<T*>(sdata);
    const int h_dim = spatial_dim; // height * width
    const int channel_idx = blockIdx.x % channels;
    const int batch_idx = blockIdx.x / channels;
    const int feature_size = spatial_dim;

    // Compute mean and variance for this channel in the batch sample
    T mean = 0, var = 0;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        int idx = batch_idx * channels * spatial_dim + channel_idx * spatial_dim + i;
        T val = input[idx];
        shared[threadIdx.x + i] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mean = 0;
        for (int i = 0; i < feature_size; i++) {
            mean += shared[i];
        }
        mean /= feature_size;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        var = 0;
        for (int i = 0; i < feature_size; i++) {
            var += (shared[i] - mean) * (shared[i] - mean);
        }
        var = var / feature_size + eps;
        var = 1.0 / sqrt(var);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        int idx = batch_idx * channels * spatial_dim + channel_idx * spatial_dim + i;
        T x_norm = (shared[i] - mean) * var;
        x_norm = x_norm * weight[channel_idx] + bias[channel_idx];
        output[idx] = x_norm / divide_by;
    }
}

torch::Tensor fused_instance_norm_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps,
    float divide_by) {

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3);
    const dim3 blocks(batch * channels);
    const dim3 threads(256); // Assuming feature size is manageable in threads

    auto output = torch::empty_like(input);

    const int shared_mem_size = sizeof(float) * spatial_dim;
    fused_instance_norm_div<float><<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        batch, channels, spatial_dim, eps, divide_by);

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in fused_instance_norm_div kernel: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

# Compile the fused kernel
fused_norm_div_cpp = (
    "torch::Tensor fused_instance_norm_div_cuda(torch::Tensor input, "
    "torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, "
    "float eps, float divide_by);"
)

fused_instance_norm = load_inline(
    name="fused_instance_norm",
    cpp_sources=fused_norm_div_cpp,
    cuda_sources=fused_norm_div_source,
    functions=["fused_instance_norm_div_cuda"],
    verbose=True,
    extra_cflags=["-D_FORCE_INLINES"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Keep track of instance norm parameters
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by
        self.fused_instance_norm = fused_instance_norm

    def forward(self, x):
        # Apply convolution normally
        x = self.conv(x)
        # Use fused kernel for InstanceNorm + division
        weight = self.instance_norm.weight
        bias = self.instance_norm.bias
        running_mean = self.instance_norm.running_mean
        running_var = self.instance_norm.running_var
        eps = self.instance_norm.eps

        x = self.fused_instance_norm.fused_instance_norm_div_cuda(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            self.divide_by,
        )
        return x

# Ensure the get_inputs and get_init_inputs are compatible with the original
# The original functions are provided, so we don't need to redefine them here.