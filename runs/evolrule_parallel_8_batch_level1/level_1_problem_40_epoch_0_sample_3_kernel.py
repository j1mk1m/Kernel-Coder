import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void layer_norm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* out,
    int batch_size,
    int features,
    int dim1,
    int dim2,
    T eps
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int D = features * dim1 * dim2;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    T* s_sum_x = reinterpret_cast<T*>(sdata);
    T* s_sum_x_sq = s_sum_x + blockDim.x;
    T* s_mean = s_sum_x_sq + blockDim.x;
    T* s_inv_std = s_mean + 1;

    s_sum_x[tid] = 0;
    s_sum_x_sq[tid] = 0;
    __syncthreads();

    for (int i = tid; i < D; i += blockDim.x) {
        int pos = batch_idx * D + i;
        T xi = x[pos];
        s_sum_x[tid] += xi;
        s_sum_x_sq[tid] += xi * xi;
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
        T sum_x = s_sum_x[0];
        T sum_x_sq = s_sum_x_sq[0];
        T mean = sum_x / D;
        T var = (sum_x_sq / D) - mean * mean;
        T inv_std = 1.0 / sqrt(var + eps);
        s_mean[0] = mean;
        s_inv_std[0] = inv_std;
    }
    __syncthreads();

    for (int i = tid; i < D; i += blockDim.x) {
        int pos = batch_idx * D + i;
        T xi = x[pos];
        int gamma_idx = i;
        T g = gamma[gamma_idx];
        T b = beta[gamma_idx];
        T normalized = (xi - s_mean[0]) * s_inv_std[0];
        out[pos] = normalized * g + b;
    }
}

at::Tensor layer_norm_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int features,
    int dim1,
    int dim2,
    float eps
) {
    auto batch_size = x.size(0);
    auto D = features * dim1 * dim2;
    auto out = at::empty_like(x);

    const int block_size = 256;
    int shared_size = (2 * block_size + 2) * sizeof(float);
    dim3 grid(batch_size);
    dim3 block(block_size);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layer_norm_cuda", ([&] {
        layer_norm_kernel<scalar_t><<<grid, block, shared_size>>>(
            x.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            features,
            dim1,
            dim2,
            eps
        );
    }));

    cudaDeviceSynchronize();
    return out;
}
"""

layer_norm_header = """
#include <torch/extension.h>
at::Tensor layer_norm_cuda(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int features,
    int dim1,
    int dim2,
    float eps
);
"""

layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_header,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        features, dim1, dim2 = normalized_shape
        D = features * dim1 * dim2
        self.weight = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.zeros(D))
        self.features = features
        self.dim1 = dim1
        self.dim2 = dim2
        self.eps = 1e-5  # Matches PyTorch's default epsilon

    def forward(self, x):
        return layer_norm.layer_norm_cuda(
            x,
            self.weight,
            self.bias,
            self.features,
            self.dim1,
            self.dim2,
            self.eps
        )