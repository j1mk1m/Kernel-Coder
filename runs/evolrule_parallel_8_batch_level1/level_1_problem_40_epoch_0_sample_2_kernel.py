import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layernorm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int batch_size,
    int F,
    int D1,
    int D2,
    float eps
) {
    int b = blockIdx.x;
    int N = F * D1 * D2;
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    // Compute sum_x
    float sum_x = 0.0f;
    for (int k = tid; k < N; k += blockDim.x) {
        int offset = b * N + k;
        sum_x += x[offset];
    }

    shared[threadIdx.x] = sum_x;
    __syncthreads();

    // Reduction step for sum_x
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float total_sum = shared[0];
    __syncthreads();

    float mean = total_sum / N;

    // Compute sum_x_sq
    float sum_x_sq = 0.0f;
    for (int k = tid; k < N; k += blockDim.x) {
        int offset = b * N + k;
        float x_val = x[offset];
        sum_x_sq += (x_val - mean) * (x_val - mean);
    }

    shared[threadIdx.x] = sum_x_sq;
    __syncthreads();

    // Reduction step for sum_x_sq
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float variance = shared[0] / N;
    float inv_std = rsqrt(variance + eps);

    __syncthreads();

    // Compute output
    for (int k = tid; k < N; k += blockDim.x) {
        int offset = b * N + k;
        float x_val = x[offset];
        float normalized = (x_val - mean) * inv_std;
        float gamma_val = gamma[k];
        float beta_val = beta[k];
        y[offset] = normalized * gamma_val + beta_val;
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int F,
    int D1,
    int D2,
    float eps
) {
    int batch_size = x.size(0);
    int N = F * D1 * D2;

    auto y = torch::empty_like(x);

    const int block_size = 256;
    int shared_size = block_size * sizeof(float);

    dim3 grid(batch_size);
    dim3 block(block_size);

    layernorm_kernel<<<grid, block, shared_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        F,
        D1,
        D2,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in layernorm_kernel: %s\\n", cudaGetErrorString(err));
    }

    return y;
}
"""

layernorm_header = """
torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int F,
    int D1,
    int D2,
    float eps
);
"""

layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layernorm_header,
    cuda_sources=layernorm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        F, D1, D2 = normalized_shape
        self.weight = nn.Parameter(torch.ones(F, D1, D2))
        self.bias = nn.Parameter(torch.zeros(F, D1, D2))
        self.F = F
        self.D1 = D1
        self.D2 = D2
        self.eps = 1e-5

    def forward(self, x):
        gamma_flat = self.weight.view(-1).contiguous()
        beta_flat = self.bias.view(-1).contiguous()
        return layer_norm.layer_norm_cuda(
            x.contiguous(),
            gamma_flat,
            beta_flat,
            self.F,
            self.D1,
            self.D2,
            self.eps
        )