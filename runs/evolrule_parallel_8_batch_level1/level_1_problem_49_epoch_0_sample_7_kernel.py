import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void max_reduction_kernel(
    const float* input_data,
    float* output_data,
    int batch_size,
    int dim1,
    int dim2,
    int reduction_dim,
    int output_size0,
    int output_size1
) {
    int output_idx = blockIdx.x;
    int out_0 = output_idx / output_size1;
    int out_1 = output_idx % output_size1;

    float max_val = -std::numeric_limits<float>::infinity();

    int reduction_size;
    if (reduction_dim == 0) {
        reduction_size = batch_size;
    } else if (reduction_dim == 1) {
        reduction_size = dim1;
    } else {
        reduction_size = dim2;
    }

    for (int i = threadIdx.x; i < reduction_size; i += blockDim.x) {
        int input_0, input_1, input_2;
        if (reduction_dim == 0) {
            input_0 = i;
            input_1 = out_0;
            input_2 = out_1;
        } else if (reduction_dim == 1) {
            input_0 = out_0;
            input_1 = i;
            input_2 = out_1;
        } else {
            input_0 = out_0;
            input_1 = out_1;
            input_2 = i;
        }

        int input_linear = input_0 * dim1 * dim2 + input_1 * dim2 + input_2;
        float val = input_data[input_linear];
        if (val > max_val) {
            max_val = val;
        }
    }

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    shared[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_data[output_idx] = shared[0];
    }
}

torch::Tensor max_reduction_cuda(torch::Tensor input, int reduction_dim) {
    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);

    int output_size0, output_size1;
    if (reduction_dim == 0) {
        output_size0 = dim1;
        output_size1 = dim2;
    } else if (reduction_dim == 1) {
        output_size0 = batch_size;
        output_size1 = dim2;
    } else {
        output_size0 = batch_size;
        output_size1 = dim1;
    }

    auto output = torch::empty({output_size0, output_size1}, input.options());

    int grid_size = output.numel();
    int block_size = 256;
    size_t shared_mem_size = block_size * sizeof(float);

    max_reduction_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2,
        reduction_dim,
        output_size0,
        output_size1
    );

    return output;
}
"""

max_reduction_cpp_source = """
torch::Tensor max_reduction_cuda(torch::Tensor input, int reduction_dim);
"""

max_reduction = load_inline(
    name="max_reduction",
    cpp_sources=max_reduction_cpp_source,
    cuda_sources=max_reduction_source,
    functions=["max_reduction_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.max_reduction = max_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda()
        return self.max_reduction.max_reduction_cuda(x, self.dim)