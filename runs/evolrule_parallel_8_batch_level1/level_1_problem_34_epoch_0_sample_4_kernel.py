import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* x, float* out,
                                    const float* gamma, const float* beta,
                                    int batch_size, int channels,
                                    int height, int width,
                                    float eps) {
    int c = blockIdx.x % channels;
    int b = blockIdx.x / channels;
    int H = height;
    int W = width;
    int N = H * W;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elements = H * W;
    int start = tid * (num_elements / num_threads);
    int end = start + (num_elements / num_threads);
    if (tid == num_threads - 1) {
        end = num_elements;
    }

    // Step 1: Compute local sums
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = start; i < end; i++) {
        int h = i / W;
        int w = i % W;
        int idx = b * channels * H * W + c * H * W + h * W + w;
        float x_val = x[idx];
        local_sum += x_val;
        local_sum_sq += x_val * x_val;
    }

    // Step 2: Reduce to block's total
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float total_sum_sq = s_sum_sq[0];

    // Compute mean and variance
    float mean = total_sum / N;
    float var = (total_sum_sq - (total_sum * total_sum) / N) / N;
    float inv_sqrt_var = rsqrtf(var + eps);

    // Broadcast mean and inv_sqrt_var to all threads
    __shared__ float s_mean, s_inv_sqrt_var;
    if (tid == 0) {
        s_mean = mean;
        s_inv_sqrt_var = inv_sqrt_var;
    }
    __syncthreads();

    // Step 3: Apply normalization and write output
    for (int i = start; i < end; i++) {
        int h = i / W;
        int w = i % W;
        int idx = b * channels * H * W + c * H * W + h * W + w;
        float x_val = x[idx];
        float normalized = (x_val - s_mean) * s_inv_sqrt_var;
        normalized *= gamma[c];
        normalized += beta[c];
        out[idx] = normalized;
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x,
                                torch::Tensor gamma,
                                torch::Tensor beta,
                                int channels,
                                int height,
                                int width,
                                float eps) {
    auto batch_size = x.size(0);
    auto out = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = batch_size * channels;
    dim3 threads(block_size);
    dim3 blocks(num_blocks);
    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size, channels, height, width,
        eps
    );
    return out;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_cuda(torch::Tensor x,
                                torch::Tensor gamma,
                                torch::Tensor beta,
                                int channels,
                                int height,
                                int width,
                                float eps);
"""

# Compile the inline CUDA code
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.gamma.device
        x = x.to(device)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        return instance_norm.instance_norm_cuda(
            x, self.gamma, self.beta, C, H, W, 1e-5
        )

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]