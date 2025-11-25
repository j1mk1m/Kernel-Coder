import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin
argmin_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmin_kernel(
    const scalar_t* input,
    int64_t* output,
    int batch_size,
    int dim1_size,
    int dim2_size,
    int reduction_dim
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size;
    if (reduction_dim == 0) {
        output_size = dim1_size * dim2_size;
    } else if (reduction_dim == 1) {
        output_size = batch_size * dim2_size;
    } else { // reduction_dim == 2
        output_size = batch_size * dim1_size;
    }

    if (out_idx >= output_size) return;

    int dim_size;
    if (reduction_dim == 0) dim_size = batch_size;
    else if (reduction_dim == 1) dim_size = dim1_size;
    else dim_size = dim2_size;

    int start, stride_step;

    if (reduction_dim == 0) {
        int d1 = out_idx / dim2_size;
        int d2 = out_idx % dim2_size;
        start = d1 * dim2_size + d2;
        stride_step = dim1_size * dim2_size;
    } else if (reduction_dim == 1) {
        int b = out_idx / dim2_size;
        int d2 = out_idx % dim2_size;
        start = b * (dim1_size * dim2_size) + d2;
        stride_step = dim2_size;
    } else {
        int b = out_idx / dim1_size;
        int d1 = out_idx % dim1_size;
        start = b * (dim1_size * dim2_size) + d1 * dim2_size;
        stride_step = 1;
    }

    int min_idx = 0;
    scalar_t min_val = input[start];

    for (int i = 1; i < dim_size; ++i) {
        scalar_t current_val = input[start + i * stride_step];
        if (current_val < min_val) {
            min_val = current_val;
            min_idx = i;
        }
    }

    output[out_idx] = min_idx;
}

torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    auto input_shape = input.sizes();
    if (input.dim() != 3) {
        throw std::invalid_argument("Input must be a 3D tensor");
    }
    int batch_size = input_shape[0];
    int dim1_size = input_shape[1];
    int dim2_size = input_shape[2];

    int output_size;
    torch::IntArrayRef output_shape;
    if (dim == 0) {
        output_size = dim1_size * dim2_size;
        output_shape = {dim1_size, dim2_size};
    } else if (dim == 1) {
        output_size = batch_size * dim2_size;
        output_shape = {batch_size, dim2_size};
    } else {
        output_size = batch_size * dim1_size;
        output_shape = {batch_size, dim1_size};
    }

    auto output = torch::empty(output_shape, torch::dtype(torch::kInt64).device(input.device()));

    const int block_size = 256;
    int num_blocks = (output_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            batch_size,
            dim1_size,
            dim2_size,
            dim
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

argmin_cpp_source = """
torch::Tensor argmin_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
argmin_cuda_mod = load_inline(
    name="argmin_cuda",
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_source,
    functions=["argmin_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return argmin_cuda_mod.argmin_cuda(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [dim]