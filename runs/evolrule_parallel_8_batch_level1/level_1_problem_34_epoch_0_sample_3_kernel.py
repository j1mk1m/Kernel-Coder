import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <tuple>

__global__ void compute_sums(
    const float* input,
    float* sum_x,
    float* sum_x_sq,
    int B, int C, int H, int W) {
    int bc_idx = blockIdx.x;
    int b = bc_idx / C;
    int c = bc_idx % C;

    int h_start = threadIdx.x;
    int stride = blockDim.x;
    float local_sum_x = 0.0f;
    float local_sum_x_sq = 0.0f;

    for (int idx = h_start; idx < H*W; idx += stride) {
        int h = idx / W;
        int w = idx % W;
        int input_offset = b * C * H * W + c * H * W + h * W + w;
        float x_val = input[input_offset];
        local_sum_x += x_val;
        local_sum_x_sq += x_val * x_val;
    }

    __shared__ float s_sum_x[256];
    __shared__ float s_sum_x_sq[256];

    int tid = threadIdx.x;
    s_sum_x[tid] = local_sum_x;
    s_sum_x_sq[tid] = local_sum_x_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_x[tid] += s_sum_x[tid + s];
            s_sum_x_sq[tid] += s_sum_x_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_x[bc_idx] = s_sum_x[0];
        sum_x_sq[bc_idx] = s_sum_x_sq[0];
    }
}

__global__ void compute_mean_var(
    const float* sum_x,
    const float* sum_x_sq,
    float* mean,
    float* var,
    int B, int C, int H, int W, float eps) {
    int bc_idx = blockIdx.x;
    int N = H * W;

    float s_x = sum_x[bc_idx];
    float s_x_sq = sum_x_sq[bc_idx];

    float m = s_x / N;
    mean[bc_idx] = m;

    float v = (s_x_sq / N) - m * m;
    var[bc_idx] = v;
}

__global__ void normalize(
    const float* input,
    const float* mean,
    float* output,
    const float* var,
    int B, int C, int H, int W,
    float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * H * W) return;

    int b = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    rem %= (H * W);
    int h = rem / W;
    int w = rem % W;

    int bc_idx = b * C + c;
    float m = mean[bc_idx];
    float v = var[bc_idx];
    float denom = 1.0f / sqrtf(v + eps);

    int input_offset = b * C * H * W + c * H * W + h * W + w;
    float x_val = input[input_offset];
    output[input_offset] = (x_val - m) * denom;
}

std::tuple<torch::Tensor, torch::Tensor> compute_sums_cuda(torch::Tensor input, int B, int C, int H, int W) {
    auto sum_x = torch::empty({B*C}, input.options());
    auto sum_x_sq = torch::empty({B*C}, input.options());

    const int block_size = 256;
    const int grid_size = B * C;

    compute_sums<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        sum_x.data_ptr<float>(),
        sum_x_sq.data_ptr<float>(),
        B, C, H, W);

    return std::make_tuple(sum_x, sum_x_sq);
}

std::tuple<torch::Tensor, torch::Tensor> compute_mean_var_cuda(
    torch::Tensor sum_x,
    torch::Tensor sum_x_sq,
    int B, int C, int H, int W,
    float eps) {
    auto mean = torch::empty({B*C}, sum_x.options());
    auto var = torch::empty({B*C}, sum_x.options());

    const int grid_size = B * C;
    const int block_size = 1;

    compute_mean_var<<<grid_size, block_size>>>(
        sum_x.data_ptr<float>(),
        sum_x_sq.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        B, C, H, W, eps);

    return std::make_tuple(mean, var);
}

torch::Tensor normalize_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    int B, int C, int H, int W,
    float eps) {
    auto output = torch::empty_like(input);

    const int num_elements = B * C * H * W;
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    normalize<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        output.data_ptr<float>(),
        var.data_ptr<float>(),
        B, C, H, W,
        eps);

    return output;
}
"""

cuda_functions = [
    "std::tuple<torch::Tensor, torch::Tensor> compute_sums_cuda(torch::Tensor input, int B, int C, int H, int W)",
    "std::tuple<torch::Tensor, torch::Tensor> compute_mean_var_cuda(torch::Tensor sum_x, torch::Tensor sum_x_sq, int B, int C, int H, int W, float eps)",
    "torch::Tensor normalize_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, int B, int C, int H, int W, float eps)"
]

module = load_inline(
    name="instance_norm_cuda",
    cuda_sources=cuda_source,
    functions=cuda_functions,
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5  # Default epsilon for InstanceNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, H, W = x.size()
        assert C == self.num_features, "Input channels must match num_features"

        # Compute sums
        sum_x, sum_x_sq = module.compute_sums_cuda(x, B, C, H, W)

        # Compute mean and var
        mean, var = module.compute_mean_var_cuda(sum_x, sum_x_sq, B, C, H, W, self.eps)

        # Normalize
        output = module.normalize_cuda(x, mean, var, B, C, H, W, self.eps)

        return output