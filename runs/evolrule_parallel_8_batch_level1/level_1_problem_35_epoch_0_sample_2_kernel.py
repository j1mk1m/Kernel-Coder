import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void group_norm_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    int B, int C, int H, int W, int G,
    float eps) {
    int b = blockIdx.x;
    int g = blockIdx.y;
    if (b >= B || g >= G) return;

    int C_per_group = C / G;
    int c_start = g * C_per_group;
    int total_elements = C_per_group * H * W;

    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sq = sdata + blockDim.x;
    float* block_mean = sdata + 2 * blockDim.x;
    float* block_inv_std = block_mean + 1;

    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    s_sq[tid] = 0.0f;
    __syncthreads();

    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int c_offset = idx / (H * W);
        int spatial_idx = idx % (H * W);
        int h = spatial_idx / W;
        int w = spatial_idx % W;
        int c = c_start + c_offset;

        int in_idx = b * C * H * W + c * H * W + h * W + w;
        scalar_t val = x[in_idx];
        s_sum[tid] += static_cast<float>(val);
        s_sq[tid] += static_cast<float>(val * val);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid] += s_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / total_elements;
        float var = s_sq[0] / total_elements - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        block_mean[0] = mean;
        block_inv_std[0] = inv_std;
    }
    __syncthreads();

    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int c_offset = idx / (H * W);
        int spatial_idx = idx % (H * W);
        int h = spatial_idx / W;
        int w = spatial_idx % W;
        int c = c_start + c_offset;

        int in_idx = b * C * H * W + c * H * W + h * W + w;
        scalar_t val = x[in_idx];
        float normed = (static_cast<float>(val) - block_mean[0]) * block_inv_std[0];
        scalar_t gamma_val = gamma[c];
        scalar_t beta_val = beta[c];
        output[in_idx] = static_cast<scalar_t>(normed * static_cast<float>(gamma_val) + static_cast<float>(beta_val));
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps) {
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int G = num_groups;

    if (C % G != 0) {
        AT_ERROR("Number of channels must be divisible by num_groups");
    }

    auto output = torch::empty_like(x);
    const int threads = 256;
    dim3 blocks(B, G);
    dim3 threadsPerBlock(threads);
    int smem_size = (2 * threads + 2) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_cuda", ([&] {
        group_norm_kernel<scalar_t><<<blocks, threadsPerBlock, smem_size, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            B, C, H, W, G, eps);
    }));

    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps);
"""

group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features, dtype=torch.float32).cuda())
        self.beta = nn.Parameter(torch.zeros(num_features, dtype=torch.float32).cuda())
        self.group_norm = group_norm

    def forward(self, x):
        x = x.contiguous()
        return self.group_norm.group_norm_cuda(
            x, self.gamma, self.beta, self.num_groups, 1e-5
        )

batch_size = 112
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features, num_groups]