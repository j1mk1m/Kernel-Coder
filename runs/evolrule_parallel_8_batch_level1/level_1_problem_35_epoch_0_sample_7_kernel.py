import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

compute_mean_var_and_group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__device__ T rsqrt(T x) {
    return 1.0f / sqrt(x);
}

extern "C" __global__ void compute_mean_var_kernel(
    const float* __restrict__ x,
    float* __restrict__ means,
    float* __restrict__ vars,
    int N, int C, int G, int H, int W, float eps) {

    int blockId = blockIdx.x;
    int n = blockId / G;
    int g = blockId % G;

    int C_g = C / G;
    int elements = C_g * H * W;

    extern __shared__ float shared_mem[];
    float* s_sum = shared_mem;
    float* s_sq_sum = shared_mem + blockDim.x;

    if (threadIdx.x == 0) {
        for (int i = 0; i < 2 * blockDim.x; ++i) {
            shared_mem[i] = 0.0f;
        }
    }
    __syncthreads();

    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;

    for (int i = threadIdx.x; i < elements; i += blockDim.x) {
        int c_local = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;

        int c_global = g * C_g + c_local;
        int offset = n * C * H * W + c_global * H * W + h * W + w;

        float val = x[offset];
        thread_sum += val;
        thread_sq_sum += val * val;
    }

    s_sum[threadIdx.x] = thread_sum;
    s_sq_sum[threadIdx.x] = thread_sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sq_sum[threadIdx.x] += s_sq_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total_sum = s_sum[0];
        float total_sq_sum = s_sq_sum[0];
        float mean = total_sum / elements;
        float var = (total_sq_sum / elements) - (mean * mean);
        means[n * G + g] = mean;
        vars[n * G + g] = var;
    }
}

extern "C" __global__ void group_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ means,
    const float* __restrict__ vars,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int G, int H, int W, float eps) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * H * W) return;

    int w = index % W;
    int rem = index / W;
    int h = rem % H;
    rem /= H;
    int c = rem % C;
    int n = rem / C;

    int C_g = C / G;
    int g = c / C_g;

    float mean = means[n * G + g];
    float var = vars[n * G + g] + eps;
    float inv_std = rsqrt(var);

    float val = x[index];
    float norm_val = (val - mean) * inv_std;

    float gamma_val = gamma[c];
    float beta_val = beta[c];

    y[index] = norm_val * gamma_val + beta_val;
}

extern "C" {
    void compute_mean_var(
        torch::Tensor x,
        torch::Tensor means,
        torch::Tensor vars,
        int N, int C, int G, int H, int W, float eps) {

        int num_blocks = N * G;
        int block_size = 256;
        int shared_mem_size = 2 * block_size * sizeof(float);

        compute_mean_var_kernel<<<num_blocks, block_size, shared_mem_size>>>(
            x.data_ptr<float>(),
            means.data_ptr<float>(),
            vars.data_ptr<float>(),
            N, C, G, H, W, eps);
        
        cudaDeviceSynchronize();
    }

    void group_norm(
        torch::Tensor x,
        torch::Tensor means,
        torch::Tensor vars,
        torch::Tensor gamma,
        torch::Tensor beta,
        torch::Tensor y,
        int N, int C, int G, int H, int W, float eps) {

        int total_elements = N * C * H * W;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        group_norm_kernel<<<grid_size, block_size>>>(
            x.data_ptr<float>(),
            means.data_ptr<float>(),
            vars.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            N, C, G, H, W, eps);

        cudaDeviceSynchronize();
    }
}
"""

cpp_source = """
extern "C" void compute_mean_var(torch::Tensor x, torch::Tensor means, torch::Tensor vars, int N, int C, int G, int H, int W, float eps);
extern "C" void group_norm(torch::Tensor x, torch::Tensor means, torch::Tensor vars, torch::Tensor gamma, torch::Tensor beta, torch::Tensor y, int N, int C, int G, int H, int W, float eps);
"""

custom_ops = load_inline(
    name="custom_group_norm",
    cpp_sources=cpp_source,
    cuda_sources=compute_mean_var_and_group_norm_source,
    functions=["compute_mean_var", "group_norm"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0, "Number of channels must be divisible by num_groups"

        means = torch.zeros(N, G, device=x.device, dtype=x.dtype)
        vars = torch.zeros(N, G, device=x.device, dtype=x.dtype)

        custom_ops.compute_mean_var(
            x, means, vars, N, C, G, H, W, 1e-5)

        out = torch.empty_like(x)

        custom_ops.group_norm(
            x, means, vars, self.gamma, self.beta, out, N, C, G, H, W, 1e-5)

        return out