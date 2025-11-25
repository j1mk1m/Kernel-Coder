import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void min_reduction_kernel(
    const float* input,
    float* output,
    int B,
    int D1,
    int D2,
    int dim
) {
    int block_idx = blockIdx.x;
    int tid = threadIdx.x;

    int num_elements, input_offset, step;
    int output_coords[3];

    if (dim == 0) {
        int d1 = block_idx / D2;
        int d2 = block_idx % D2;
        output_coords[0] = d1;
        output_coords[1] = d2;
        input_offset = d1 * D2 + d2;
        step = D1 * D2;
        num_elements = B;
    } else if (dim == 1) {
        int b = block_idx / D2;
        int d2 = block_idx % D2;
        output_coords[0] = b;
        output_coords[1] = d2;
        input_offset = b * D1 * D2 + d2;
        step = D2;
        num_elements = D1;
    } else {
        int b = block_idx / D1;
        int d1 = block_idx % D1;
        output_coords[0] = b;
        output_coords[1] = d1;
        input_offset = b * D1 * D2 + d1 * D2;
        step = 1;
        num_elements = D2;
    }

    extern __shared__ float shared_data[];
    for (int i = tid; i < num_elements; i += blockDim.x) {
        shared_data[i] = input[input_offset + i * step];
    }
    __syncthreads();

    int segment_size = (num_elements + blockDim.x - 1) / blockDim.x;
    int start = tid * segment_size;
    int end = min(start + segment_size, num_elements);
    float local_min = std::numeric_limits<float>::max();
    for (int i = start; i < end; ++i) {
        if (shared_data[i] < local_min) {
            local_min = shared_data[i];
        }
    }

    __shared__ float partial_min[256];
    partial_min[tid] = local_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (partial_min[tid] > partial_min[tid + s]) {
                partial_min[tid] = partial_min[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int output_offset;
        if (dim == 0) {
            output_offset = d1 * D2 + d2;
        } else if (dim == 1) {
            output_offset = b * D2 + d2;
        } else {
            output_offset = b * D1 + d1;
        }
        output[output_offset] = partial_min[0];
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    auto input_contig = input.contiguous();
    int B = input_contig.size(0);
    int D1 = input_contig.size(1);
    int D2 = input_contig.size(2);

    std::vector<int64_t> output_shape;
    if (dim == 0) {
        output_shape = {D1, D2};
    } else if (dim == 1) {
        output_shape = {B, D2};
    } else {
        output_shape = {B, D1};
    }
    auto output = torch::empty(output_shape, input.options());

    int block_size = 256;
    int grid_size = 1;
    for (int s : output_shape) {
        grid_size *= s;
    }

    int num_elements;
    if (dim == 0) {
        num_elements = B;
    } else if (dim == 1) {
        num_elements = D1;
    } else {
        num_elements = D2;
    }

    int shared_mem_size = num_elements * sizeof(float);

    min_reduction_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2, dim
    );

    return output;
}
"""

min_reduction_cpp_source = (
    "torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);"
)

min_reduction = load_inline(
    name="min_reduction",
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=["min_reduction_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)