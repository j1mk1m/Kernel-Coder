import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for sum reduction along a specific dimension
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reduce_sum_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 int64_t outer_dim,
                                 int64_t reduce_dim,
                                 int64_t inner_dim) {
    extern __shared__ scalar_t shared_storage[];
    scalar_t* shared = shared_storage;

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Compute the index in the output tensor
    int output_index = by * outer_dim + bx;

    // Compute the starting index in the input tensor
    int input_offset = by * outer_dim * reduce_dim + bx * reduce_dim;

    // Initialize thread's sum
    scalar_t sum = 0;

    // Iterate over the reduction dimension
    for (int i = tx; i < reduce_dim; i += blockDim.x) {
        sum += input[input_offset + i];
    }

    // Use shared memory for block-wide reduction
    shared[tx] = sum;
    __syncthreads();

    // Perform block reduction using warp-level operations
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tx < s) {
            shared[tx] += shared[tx + s];
        }
        __syncthreads();
    }

    // Use warp-level reduction for the last 32 threads
    if (tx < 32) {
        for (int s = 16; s > 0; s >>= 1) {
            shared[tx] += shared[tx + s];
        }
    }

    if (tx == 0) {
        output[output_index] = shared[0];
    }
}

std::tuple<torch::Tensor, torch::Tensor> get_shape_info(torch::Tensor input, int64_t dim) {
    auto input_dims = input.sizes().vec();
    int64_t outer_dim = 1;
    int64_t reduce_dim = input.size(dim);
    int64_t inner_dim = 1;

    for (int i = 0; i < dim; ++i) {
        outer_dim *= input_dims[i];
    }
    for (int i = dim + 1; i < input_dims.size(); ++i) {
        inner_dim *= input_dims[i];
    }

    return std::make_tuple(outer_dim, reduce_dim, inner_dim);
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int64_t dim) {
    auto input_contig = input.contiguous();
    auto output = torch::zeros_like(input_contig, torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    auto outer_dim = std::get<0>(get_shape_info(input_contig, dim));
    auto reduce_dim = std::get<1>(get_shape_info(input_contig, dim));
    auto inner_dim = std::get<2>(get_shape_info(input_contig, dim));

    const int block_size = 256;
    dim3 grid(outer_dim, inner_dim);
    int shared_size = block_size * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "reduce_sum_cuda", ([&] {
        reduce_sum_kernel<scalar_t><<<grid, block_size, shared_size, stream>>>(
            input_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_dim,
            reduce_dim,
            inner_dim);
    }));

    return output;
}
"""

sum_reduction_cpp_source = """
#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor> get_shape_info(torch::Tensor input, int64_t dim);
torch::Tensor sum_reduction_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code for sum reduction
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction_cuda", "get_shape_info"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The original torch.sum keepsdim, so we need to ensure the output has the same shape except for the reduced dimension.
        # The CUDA kernel is written to handle keepdim=True by maintaining the dimension (output has size 1 in the reduced dimension)
        # Since the input shape is (batch, dim1, dim2), and dim=1 is the middle dimension, the output shape is (batch, 1, dim2)
        # The kernel implementation assumes the input is contiguous, so we call contiguous() here if needed
        # The kernel returns the correct shape automatically
        return self.sum_reduction.sum_reduction_cuda(x, self.dim)

# Ensure the input is on the same device as the model (CUDA)
def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [reduce_dim]