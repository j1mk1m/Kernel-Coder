import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_sources = """
at::Tensor exclusive_cumsum_cuda(
    at::Tensor input,
    int dim);
"""

cuda_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void exclusive_cumsum_kernel(
    const float* input,
    float* output,
    int batch_size,
    int dim_size,
    int dim) 
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const int T = (dim_size + B - 1) / B;

    __shared__ float seg_totals[B];

    float seg_total = 0.0f;
    float seg_cumsum = 0.0f;

    for (int local_i = 0; local_i < T; ++local_i) {
        int global_i = tid * T + local_i;
        if (global_i >= dim_size)
            break;

        float val = input[row * dim_size + global_i];

        if (local_i == 0) {
            output[row * dim_size + global_i] = 0.0f;
            seg_cumsum = 0.0f;
            seg_total += val;
        } else {
            output[row * dim_size + global_i] = seg_cumsum;
            seg_cumsum += val;
            seg_total += val;
        }
    }

    seg_totals[tid] = seg_total;
    __syncthreads();

    for (int stride = 1; stride <= B; stride *= 2) {
        __syncthreads();
        if (tid >= stride) {
            seg_totals[tid] += seg_totals[tid - stride];
        }
    }

    float exclusive_prefix;
    if (tid == 0) {
        exclusive_prefix = 0.0f;
    } else {
        exclusive_prefix = seg_totals[tid - 1];
    }
    __syncthreads();

    for (int local_i = 0; local_i < T; ++local_i) {
        int global_i = tid * T + local_i;
        if (global_i >= dim_size)
            break;

        output[row * dim_size + global_i] += exclusive_prefix;
    }
}

at::Tensor exclusive_cumsum_cuda(
    at::Tensor input,
    int dim) 
{
    const int batch_size = input.size(0);
    const int dim_size = input.size(dim);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    dim3 blocks(batch_size);
    dim3 threads(block_size);
    size_t shared_size = block_size * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    exclusive_cumsum_kernel<<<blocks, threads, shared_size, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size,
        dim
    );

    return output;
}
"""

exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=cpp_sources,
    cuda_sources=cuda_sources,
    functions=["exclusive_cumsum_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return exclusive_cumsum.exclusive_cumsum_cuda(x, self.dim)