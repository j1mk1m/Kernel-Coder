import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    float* __restrict__ out,
    int batch_size,
    int F,
    int D1,
    int D2,
    int N
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_squares[BLOCK_SIZE];

    float local_sum = 0.0f;
    float local_squares = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        int global_x_idx = batch_idx * N + i;
        float xi = x[global_x_idx];
        local_sum += xi;
        local_squares += xi * xi;
    }

    s_sum[tid] = local_sum;
    s_squares[tid] = local_squares;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_squares[tid] += s_squares[tid + s];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float total_squares = s_squares[0];

    float mean = total_sum / N;
    float var = total_squares / N - mean * mean;
    float std = rsqrtf(var + eps);

    for (int i = tid; i < N; i += blockDim.x) {
        int global_out_idx = batch_idx * N + i;

        int f = i / (D1 * D2);
        int rem = i % (D1 * D2);
        int d1 = rem / D2;
        int d2 = rem % D2;

        int w_idx = f * D1 * D2 + d1 * D2 + d2;
        float g = weight[w_idx];
        float b = bias[w_idx];

        float xi = x[global_out_idx];
        float normalized = (xi - mean) * std;
        out[global_out_idx] = normalized * g + b;
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    int F,
    int D1,
    int D2
) {
    int B = x.size(0);
    int N = F * D1 * D2;

    auto out = torch::empty_like(x);

    dim3 blocks(B);
    dim3 threads(BLOCK_SIZE);

    layer_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        eps,
        out.data_ptr<float>(),
        B,
        F,
        D1,
        D2,
        N
    );

    return out;
}
"""

layer_norm_cpp_source = """
torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps,
    int F,
    int D1,
    int D2
);
"""

layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        F, D1, D2 = normalized_shape
        self.weight = nn.Parameter(torch.ones(F, D1, D2).cuda())
        self.bias = nn.Parameter(torch.zeros(F, D1, D2).cuda())
        self.eps = 1e-5  # Default epsilon from PyTorch LayerNorm

    def forward(self, x):
        x = x.contiguous()
        F, D1, D2 = self.normalized_shape
        return layer_norm.layer_norm_cuda(
            x, self.weight, self.bias, self.eps, F, D1, D2
        )