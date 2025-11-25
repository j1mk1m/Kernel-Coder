import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation for GroupNorm
groupnorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_mean_var_kernel(
    const float* input,
    float* means,
    float* vars,
    int N, int C, int H, int W,
    int num_groups
) {
    int g = blockIdx.x;
    int n = blockIdx.y;
    int G = num_groups;
    int C_per_group = C / G;
    int count = C_per_group * H * W;
    int c_start = g * C_per_group;

    float sum_x = 0.0f;
    float sum_x2 = 0.0f;

    for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
        int c_offset = idx / (H * W);
        int hw_idx = idx % (H * W);
        int h = hw_idx / W;
        int w = hw_idx % W;
        int c = c_start + c_offset;

        int input_offset = n * C * H * W + c * H * W + h * W + w;
        float x = input[input_offset];
        sum_x += x;
        sum_x2 += x * x;
    }

    __shared__ float shared_sum_x[256];
    __shared__ float shared_sum_x2[256];
    int tid = threadIdx.x;
    shared_sum_x[tid] = sum_x;
    shared_sum_x2[tid] = sum_x2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum_x[tid] += shared_sum_x[tid + s];
            shared_sum_x2[tid] += shared_sum_x2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = shared_sum_x[0] / count;
        float var = (shared_sum_x2[0] / count) - (mean * mean);
        int mean_var_idx = n * G + g;
        means[mean_var_idx] = mean;
        vars[mean_var_idx] = var;
    }
}

__global__ void normalize_kernel(
    const float* input,
    float* output,
    const float* means,
    const float* vars,
    const float* gamma,
    const float* beta,
    int N, int C, int H, int W,
    int num_groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;

    int n = idx / (C * H * W);
    int remainder = idx % (C * H * W);
    int c = remainder / (H * W);
    int hw_idx = remainder % (H * W);
    int h = hw_idx / W;
    int w = hw_idx % W;

    int G = num_groups;
    int C_per_group = C / G;
    int g = c / C_per_group;
    int mean_var_idx = n * G + g;

    float mean = means[mean_var_idx];
    float var = vars[mean_var_idx];
    float denom = rsqrt(var + eps);

    int input_offset = n * C * H * W + c * H * W + h * W + w;
    float x = input[input_offset];
    float normalized = (x - mean) * denom;
    float out_val = gamma[c] * normalized + beta[c];
    output[input_offset] = out_val;
}

extern "C" {
    torch::Tensor groupnorm_cuda(
        torch::Tensor input,
        torch::Tensor gamma,
        torch::Tensor beta,
        int num_groups,
        float eps
    ) {
        TORCH_CHECK(input.dim() == 4, "Input must be 4D");
        int N = input.size(0);
        int C = input.size(1);
        int H = input.size(2);
        int W = input.size(3);
        int G = num_groups;
        TORCH_CHECK(C % G == 0, "Number of channels must be divisible by num_groups");

        auto output = at::empty_like(input);
        auto means = at::empty({N * G}, input.options());
        auto vars = at::empty({N * G}, input.options());

        dim3 block(256);
        dim3 grid(G, N);
        compute_mean_var_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            means.data_ptr<float>(),
            vars.data_ptr<float>(),
            N, C, H, W,
            G
        );

        int total_elements = N * C * H * W;
        dim3 block_norm(256);
        dim3 grid_norm((total_elements + block_norm.x - 1) / block_norm.x, 1);
        normalize_kernel<<<grid_norm, block_norm>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            means.data_ptr<float>(),
            vars.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            N, C, H, W,
            G,
            eps
        );

        cudaDeviceSynchronize();
        return output;
    }
}
"""

# Load the CUDA code
groupnorm_module = load_inline(
    name="groupnorm_cuda",
    cuda_sources=groupnorm_cuda_source,
    functions=["groupnorm_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.eps = 1e-5  # Default epsilon value
        self.gamma = nn.Parameter(torch.empty(num_features))
        self.beta = nn.Parameter(torch.empty(num_features))
        # Initialize gamma and beta similar to PyTorch's GroupNorm
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return groupnorm_module.groupnorm_cuda(x, self.gamma, self.beta, self.num_groups, self.eps)