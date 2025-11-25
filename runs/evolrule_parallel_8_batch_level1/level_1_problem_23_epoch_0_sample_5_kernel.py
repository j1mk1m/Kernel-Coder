import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

template <typename scalar_t>
__global__ void softmax_kernel(const scalar_t* __restrict__ input,
                              scalar_t* __restrict__ output,
                              int batch_size,
                              int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float s_data[];

    // Phase 1: Compute max for this row
    float local_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = input[row * dim + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    s_data[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    float row_max = s_data[0];
    __syncthreads();

    // Phase 2: Compute exponentials and accumulate sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = input[row * dim + i] - row_max;
        float exp_val = expf(val);
        local_sum += exp_val;
    }

    s_data[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    float row_sum = s_data[0];
    __syncthreads();

    // Phase 3: Compute final values
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = input[row * dim + i] - row_max;
        float exp_val = expf(val);
        output[row * dim + i] = exp_val * inv_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int dim = input.size(1);

    auto output = torch::empty_like(input);

    const int block_size = 1024;
    const int shared_size = block_size * sizeof(float);

    dim3 grid(batch_size);
    dim3 block(block_size);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax_cuda", ([&] {
        softmax_kernel<scalar_t><<<grid, block, shared_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));

    return output;
}
"""

softmax_kernel_cpp_source = """
#include <torch/extension.h>

torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the CUDA code
softmax_cuda = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_kernel_cpp_source,
    cuda_sources=softmax_kernel_source,
    functions=["softmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda = softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda(x)