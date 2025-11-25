import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batch normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_forward_kernel(
    const float* input, 
    const float* running_mean, 
    const float* running_var, 
    const float* weight, 
    const float* bias, 
    float* output, 
    float eps, 
    int N, 
    int C, 
    int H, 
    int W) {
    
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    if (n >= N || c >= C) return;

    float sum = 0.0f, sum_sqr = 0.0f;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = ((n * C + c) * H + h) * W + w;
            sum += input[idx];
            sum_sqr += input[idx] * input[idx];
        }
    }

    __shared__ float shared_sum[256], shared_sum_sqr[256];
    shared_sum[threadIdx.y] = sum;
    shared_sum_sqr[threadIdx.y] = sum_sqr;
    __syncthreads();

    for (int s = blockDim.y / 2; s > 0; s /= 2) {
        if (threadIdx.y < s) {
            shared_sum[threadIdx.y] += shared_sum[threadIdx.y + s];
            shared_sum_sqr[threadIdx.y] += shared_sum_sqr[threadIdx.y + s];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        float mean = shared_sum[0] / (N * H * W);
        float var = shared_sum_sqr[0] / (N * H * W) - mean * mean;

        __shared__ float shared_running_mean[256], shared_running_var[256];
        shared_running_mean[threadIdx.z] = running_mean[c];
        shared_running_var[threadIdx.z] = running_var[c];
        __syncthreads();

        for (int s = blockDim.z / 2; s > 0; s /= 2) {
            if (threadIdx.z < s) {
                shared_running_mean[threadIdx.z] += shared_running_mean[threadIdx.z + s];
                shared_running_var[threadIdx.z] += shared_running_var[threadIdx.z + s];
            }
            __syncthreads();
        }

        if (threadIdx.z == 0) {
            atomicAdd(&running_mean[c], mean);
            atomicAdd(&running_var[c], var);
        }
    }

    __syncthreads();

    int idx = ((n * C + c) * H + h) * W + w;
    float normalized_value = (input[idx] - shared_running_mean[0]) / sqrt(shared_running_var[0] + eps);
    output[idx] = weight[c] * normalized_value + bias[c];
}

torch::Tensor batch_norm_forward_cuda(
    torch::Tensor input, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float eps) {

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::zeros_like(input);

    const int block_size = 16;
    dim3 grid(N, block_size, block_size);
    dim3 block(block_size, block_size, block_size);

    batch_norm_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        running_mean.data_ptr<float>(), 
        running_var.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        eps, 
        N, 
        C, 
        H, 
        W);

    return output;
}
"""

batch_norm_cpp_source = (
    "torch::Tensor batch_norm_forward_cuda("
    "torch::Tensor input, "
    "torch::Tensor running_mean, "
    "torch::Tensor running_var, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "float eps);"
)

# Compile the inline CUDA code for batch normalization
batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.running_mean = nn.Parameter(torch.zeros(num_features))
        self.running_var = nn.Parameter(torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return batch_norm.batch_norm_forward_cuda(
            x, 
            self.running_mean, 
            self.running_var, 
            self.weight, 
            self.bias, 
            eps=1e-5
        )