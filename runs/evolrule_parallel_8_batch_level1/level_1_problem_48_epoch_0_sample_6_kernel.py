import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code for mean reduction
mean_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(float* input, float* output,
    int N, int D1, int D2, int dim, int S, int total_out) {
    int out_idx = blockIdx.x;
    if (out_idx >= total_out) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    __syncthreads();

    int d0_out, d1_out, d2_out;
    switch (dim) {
        case 0:
            d1_out = out_idx / D2;
            d2_out = out_idx % D2;
            break;
        case 1:
            d0_out = out_idx / D2;
            d2_out = out_idx % D2;
            break;
        case 2:
            d0_out = out_idx / D1;
            d1_out = out_idx % D1;
            break;
    }

    int total_threads = blockDim.x;
    int chunk_size = (S + total_threads - 1) / total_threads;
    for (int d_dim = tid * chunk_size; d_dim < S; d_dim += total_threads) {
        int input_idx;
        if (dim == 0) {
            input_idx = d_dim * D1 * D2 + d1_out * D2 + d2_out;
        } else if (dim == 1) {
            input_idx = d0_out * D1 * D2 + d_dim * D2 + d2_out;
        } else {
            input_idx = d0_out * D1 * D2 + d1_out * D2 + d_dim;
        }
        sdata[tid] += input[input_idx];
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = sdata[0] / static_cast<float>(S);
    }
}

torch::Tensor mean_cuda(torch::Tensor input, int dim) {
    auto input_ = input.contiguous();
    auto input_data = input_.data_ptr<float>();
    int N = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    int S = input.size(dim);

    int64_t output_shape[2];
    int total_out;
    if (dim == 0) {
        output_shape[0] = D1;
        output_shape[1] = D2;
        total_out = D1 * D2;
    } else if (dim == 1) {
        output_shape[0] = N;
        output_shape[1] = D2;
        total_out = N * D2;
    } else {
        output_shape[0] = N;
        output_shape[1] = D1;
        total_out = N * D1;
    }

    auto output = torch::empty({output_shape[0], output_shape[1]}, input.options());
    auto output_data = output.data_ptr<float>();

    const int block_size = 256;
    const int num_blocks = total_out;
    mean_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input_data, output_data, N, D1, D2, dim, S, total_out
    );
    cudaDeviceSynchronize();
    return output;
}

extern "C" {
    torch::Tensor mean_cuda(torch::Tensor input, int dim);
}
"""

mean_cuda_cpp_source = (
    "torch::Tensor mean_cuda(torch::Tensor input, int dim);"
)

# Compile the CUDA code
mean_cuda = load_inline(
    name="mean_cuda",
    cpp_sources=mean_cuda_cpp_source,
    cuda_sources=mean_cuda_source,
    functions=["mean_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_cuda.mean_cuda(x, self.dim)