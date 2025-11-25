import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void compute_mean_var_kernel(
    const float* x, int batch_size, int num_channels, int spatial_size,
    float* means, float* vars, float eps) {
    int c = blockIdx.x;
    if (c >= num_channels) return;

    int elements_per_channel = batch_size * spatial_size;
    int tid = threadIdx.x;
    __shared__ float s_sum_x[BLOCK_SIZE];
    __shared__ float s_sum_x_sq[BLOCK_SIZE];

    s_sum_x[tid] = 0.0f;
    s_sum_x_sq[tid] = 0.0f;
    __syncthreads();

    for (int i = tid; i < elements_per_channel; i += blockDim.x) {
        int index = c * batch_size * spatial_size + i;
        float val = x[index];
        s_sum_x[tid] += val;
        s_sum_x_sq[tid] += val * val;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_x[tid] += s_sum_x[tid + s];
            s_sum_x_sq[tid] += s_sum_x_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum_x = s_sum_x[0];
        float sum_x_sq = s_sum_x_sq[0];
        float mean = sum_x / elements_per_channel;
        float var = (sum_x_sq / elements_per_channel) - (mean * mean);
        means[c] = mean;
        vars[c] = var;
    }
}

__global__ void batch_norm_forward_kernel(
    const float* x, const float* means, const float* vars,
    const float* gamma, const float* beta,
    float* out,
    int num_channels, int spatial_size, int batch_size, float eps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * num_channels * spatial_size) return;

    int c = (index / (batch_size * spatial_size)) % num_channels;
    float x_val = x[index];
    float mean = means[c];
    float var = vars[c];
    float denom = sqrt(var + eps);
    out[index] = ((x_val - mean) / denom) * gamma[c] + beta[c];
}

torch::Tensor batch_norm_forward_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps=1e-5) {

    int batch_size = x.size(0);
    int num_channels = x.size(1);
    int h = x.size(2);
    int w = x.size(3);
    int spatial_size = h * w;
    int total_elements = x.numel();

    auto means = torch::empty({num_channels}, device=x.device(), dtype=torch::kFloat32);
    auto vars_ = torch::empty({num_channels}, device=x.device(), dtype=torch::kFloat32);
    auto out = torch::empty_like(x);

    dim3 blocks(num_channels);
    dim3 threads(BLOCK_SIZE);
    compute_mean_var_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), batch_size, num_channels, spatial_size,
        means.data_ptr<float>(), vars_.data_ptr<float>(), eps
    );

    int num_blocks_norm = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batch_norm_forward_kernel<<<num_blocks_norm, BLOCK_SIZE>>>(
        x.data_ptr<float>(), means.data_ptr<float>(), vars_.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        out.data_ptr<float>(),
        num_channels, spatial_size, batch_size, eps
    );

    return out;
}
"""

CUDA_HEADER = """
torch::Tensor batch_norm_forward_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps=1e-5);
"""

batch_norm_cuda = load_inline(
    name="batch_norm_cuda",
    cpp_sources=CUDA_HEADER,
    cuda_sources=CUDA_SOURCE,
    functions=["batch_norm_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x):
        return batch_norm_cuda.batch_norm_forward_cuda(x, self.gamma, self.beta, self.eps)