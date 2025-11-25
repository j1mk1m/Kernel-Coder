import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <int block_size>
__global__ void l2_norm_fusion_kernel(
    const float* input,
    float* output,
    int batch_size,
    int dim
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_partials[block_size];
    __shared__ float inv_norm;

    float sum = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        float val = input[row * dim + i];
        sum += val * val;
    }

    s_partials[tid] = sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partials[tid] += s_partials[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = s_partials[0];
        inv_norm = rsqrtf(total_sum);
        s_partials[0] = inv_norm;
    }
    __syncthreads();

    inv_norm = s_partials[0];
    for (int i = tid; i < dim; i += block_size) {
        float val = input[row * dim + i];
        output[row * dim + i] = val * inv_norm;
    }
}

torch::Tensor l2_norm_fusion_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);

    l2_norm_fusion_kernel<block_size><<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}
"""

l2_norm_fusion_cpp_source = """
torch::Tensor l2_norm_fusion_cuda(torch::Tensor input);
"""

l2_norm_fusion = load_inline(
    name="l2_norm_fusion",
    cpp_sources=l2_norm_fusion_cpp_source,
    cuda_sources=l2_norm_fusion_source,
    functions=["l2_norm_fusion_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_fusion = l2_norm_fusion

    def forward(self, x):
        return self.l2_fusion.l2_norm_fusion_cuda(x)