import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void softmax_forward_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      int batch_size,
                                      int dim) {
    int batch_idx = blockIdx.x;

    __shared__ float s_max[1024];
    __shared__ float s_sum[1024];

    // Compute local max
    float local_max = -FLT_MAX;
    int index = threadIdx.x;
    while (index < dim) {
        scalar_t val = input[batch_idx * dim + index];
        if (val > local_max) {
            local_max = val;
        }
        index += blockDim.x;
    }
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce to get global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_max[threadIdx.x] < s_max[threadIdx.x + s]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float global_max = s_max[0];
    __syncthreads();

    // Compute local sum of exp(x - global_max)
    float local_sum = 0.0f;
    index = threadIdx.x;
    while (index < dim) {
        scalar_t val = input[batch_idx * dim + index];
        float exp_val = expf(static_cast<float>(val) - global_max);
        local_sum += exp_val;
        index += blockDim.x;
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce to get global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float global_sum = s_sum[0];
    __syncthreads();

    // Compute final values and write to output
    index = threadIdx.x;
    while (index < dim) {
        scalar_t val = input[batch_idx * dim + index];
        float exp_val = expf(static_cast<float>(val) - global_max);
        output[batch_idx * dim + index] = static_cast<scalar_t>(exp_val / global_sum);
        index += blockDim.x;
    }
}

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim = input.size(1);

    torch::Tensor output = torch::empty_like(input);

    const int threads = 1024;
    const int blocks = batch_size;

    softmax_forward_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}
"""

softmax_cpp = "torch::Tensor softmax_forward_cuda(torch::Tensor input);"

softmax_module = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp,
    cuda_sources=softmax_kernel,
    functions=["softmax_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_forward_cuda(x)