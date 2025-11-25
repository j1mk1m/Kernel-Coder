import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void min_reduction_kernel(const float* input, float* output, int B, int D1, int D2, int reduce_dim) {
    int base_offset, size, step;

    if (reduce_dim == 0) {
        int d1 = blockIdx.x / D2;
        int d2 = blockIdx.x % D2;
        base_offset = d1 * D2 + d2;
        size = B;
        step = D1 * D2;
    } else if (reduce_dim == 1) {
        int b = blockIdx.x / D2;
        int d2 = blockIdx.x % D2;
        base_offset = b * D1 * D2 + d2;
        size = D1;
        step = D2;
    } else if (reduce_dim == 2) {
        int b = blockIdx.x / D1;
        int d1 = blockIdx.x % D1;
        base_offset = b * D1 * D2 + d1 * D2;
        size = D2;
        step = 1;
    }

    extern __shared__ float shared_min[];
    int tid = threadIdx.x;

    float min_val = std::numeric_limits<float>::max();

    for (int i = tid; i < size; i += blockDim.x) {
        int idx = base_offset + i * step;
        float val = input[idx];
        if (val < min_val) {
            min_val = val;
        }
    }

    shared_min[tid] = min_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_min[tid] > shared_min[tid + s]) {
                shared_min[tid] = shared_min[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_min[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int reduce_dim) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    torch::IntArrayRef output_size;
    if (reduce_dim == 0) {
        output_size = {D1, D2};
    } else if (reduce_dim == 1) {
        output_size = {B, D2};
    } else {
        output_size = {B, D1};
    }

    auto output = torch::empty(output_size, input.options());

    int block_size = 256;
    int num_blocks = 0;
    if (reduce_dim == 0) {
        num_blocks = D1 * D2;
    } else if (reduce_dim == 1) {
        num_blocks = B * D2;
    } else {
        num_blocks = B * D1;
    }

    min_reduction_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2, reduce_dim
    );

    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int reduce_dim);
"""

min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)