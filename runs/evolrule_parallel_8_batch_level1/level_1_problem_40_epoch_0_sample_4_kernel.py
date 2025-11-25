import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int C,
    int H,
    int W,
    float eps) {
    const int D = C * H * W;
    const int sample_idx = blockIdx.x;
    const int sample_offset = sample_idx * D;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const int idx = sample_offset + i;
        float x = input[idx];
        local_sum += x;
        local_sum_sq += x * x;
    }

    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float total_sum_sq = s_sum_sq[0];

    float mean = total_sum / D;
    float var = total_sum_sq / D - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);

    if (threadIdx.x == 0) {
        s_sum[0] = mean;
        s_sum_sq[0] = inv_std;
    }
    __syncthreads();

    mean = s_sum[0];
    inv_std = s_sum_sq[0];

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const int idx = sample_offset + i;
        float x = input[idx];
        float normalized = (x - mean) * inv_std;
        normalized = normalized * gamma[i] + beta[i];
        output[idx] = normalized;
    }
}

torch::Tensor layernorm_cuda(torch::Tensor input,
                             torch::Tensor gamma,
                             torch::Tensor beta,
                             int batch_size,
                             int C,
                             int H,
                             int W,
                             float eps) {
    const int D = C * H * W;
    const dim3 blocks(batch_size);
    const int threads = 256;
    auto output = torch::empty_like(input);

    if (input.scalar_type() != torch::kFloat32) {
        AT_ERROR("Only float32 is supported.");
    }

    layernorm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        C,
        H,
        W,
        eps
    );

    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_cuda(torch::Tensor input,
                             torch::Tensor gamma,
                             torch::Tensor beta,
                             int batch_size,
                             int C,
                             int H,
                             int W,
                             float eps);
"""

layernorm = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=["layernorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.C, self.H, self.W = normalized_shape
        self.weight = nn.Parameter(torch.ones(self.C, self.H, self.W).cuda())
        self.bias = nn.Parameter(torch.zeros(self.C, self.H, self.W).cuda())
        self.layernorm_cuda = layernorm.layernorm_cuda

    def forward(self, x):
        batch_size = x.size(0)
        return self.layernorm_cuda(
            x.contiguous(),
            self.weight.view(-1),
            self.bias.view(-1),
            batch_size,
            self.C,
            self.H,
            self.W,
            1e-5
        )