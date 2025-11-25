import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void layer_norm_kernel(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int batch_size,
    int features,
    int dim1,
    int dim2,
    float eps
) {
    int batch_idx = blockIdx.x;
    int N = features * dim1 * dim2;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ float shared[];
    float* s_sum = &shared[0];
    float* s_sq = &shared[num_threads];

    // Compute mean
    {
        float my_sum = 0.0f;
        for (int i = tid; i < N; i += num_threads) {
            int f = i / (dim1 * dim2);
            int rem = i % (dim1 * dim2);
            int d1 = rem / dim2;
            int d2 = rem % dim2;

            int x_offset = batch_idx * features * dim1 * dim2 + f * dim1 * dim2 + d1 * dim2 + d2;
            my_sum += x[x_offset];
        }
        s_sum[tid] = my_sum;
        __syncthreads();

        for (int s = num_threads / 2; s > 0; s >>= 1) {
            if (tid < s) s_sum[tid] += s_sum[tid + s];
            __syncthreads();
        }
        float total_sum = s_sum[0];
        float mean = total_sum / N;

        // Compute variance and output
        {
            float my_sq = 0.0f;
            for (int i = tid; i < N; i += num_threads) {
                int f = i / (dim1 * dim2);
                int rem = i % (dim1 * dim2);
                int d1 = rem / dim2;
                int d2 = rem % dim2;

                int x_offset = batch_idx * features * dim1 * dim2 + f * dim1 * dim2 + d1 * dim2 + d2;
                float val = x[x_offset] - mean;
                my_sq += val * val;
            }
            s_sq[tid] = my_sq;
            __syncthreads();

            for (int s = num_threads / 2; s > 0; s >>= 1) {
                if (tid < s) s_sq[tid] += s_sq[tid + s];
                __syncthreads();
            }
            float total_sq = s_sq[0];
            float var = total_sq / N;
            float inv_std = 1.0f / sqrt(var + eps);

            for (int i = tid; i < N; i += num_threads) {
                int f = i / (dim1 * dim2);
                int rem = i % (dim1 * dim2);
                int d1 = rem / dim2;
                int d2 = rem % dim2;

                int x_offset = batch_idx * features * dim1 * dim2 + f * dim1 * dim2 + d1 * dim2 + d2;
                float val = x[x_offset];

                int gamma_beta_idx = f * dim1 * dim2 + d1 * dim2 + d2;
                float norm_val = (val - mean) * inv_std;
                out[x_offset] = norm_val * gamma[gamma_beta_idx] + beta[gamma_beta_idx];
            }
        }
    }
}

void layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor out,
                     int batch_size, int features, int dim1, int dim2, float eps) {
    const int block_size = 256;
    const int shared_mem = 2 * block_size * sizeof(float);
    const int num_blocks = batch_size;
    layer_norm_kernel<<<num_blocks, block_size, shared_mem>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        out.data_ptr<float>(), batch_size, features, dim1, dim2, eps);
}
"""

layer_norm_cpp_source = """
void layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor out,
                     int batch_size, int features, int dim1, int dim2, float eps);
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
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*normalized_shape))
        self.bias = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, features, dim1, dim2 = x.shape
        out = torch.empty_like(x)
        layer_norm.layer_norm_cuda(
            x,
            self.weight,
            self.bias,
            out,
            batch_size,
            features,
            dim1,
            dim2,
            1e-5
        )
        return out

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]