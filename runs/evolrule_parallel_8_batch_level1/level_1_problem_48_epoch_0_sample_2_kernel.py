import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int BlockSize>
__global__ void mean_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim,
    const int* __restrict__ input_dims,
    const int* __restrict__ output_dims,
    const int* __restrict__ input_strides,
    int reduction_size,
    int num_dims,
    int total_out
) {
    int out_idx = blockIdx.x;
    if (out_idx >= total_out) return;

    // Compute the base input index where reduction dim is 0
    int in_base = 0;
    int remaining = out_idx;
    int out_dim_cursor = 0;

    for (int i = 0; i < num_dims; ++i) {
        if (i == dim) continue;
        int current_out_size = output_dims[out_dim_cursor];
        int current_out_index = remaining % current_out_size;
        remaining /= current_out_size;
        out_dim_cursor += 1;

        if (i < dim) {
            in_base += current_out_index * input_strides[i];
        } else {
            in_base += current_out_index * input_strides[i];
        }
    }

    // Determine thread's portion of the reduction dimension
    int tid = threadIdx.x;
    int chunk_size = (reduction_size + BlockSize - 1) / BlockSize;
    int start = tid * chunk_size;
    int end = start + chunk_size;
    if (end > reduction_size) end = reduction_size;

    float sum = 0.0f;
    for (int pos = start; pos < end; ++pos) {
        int in_pos_idx = in_base + pos * input_strides[dim];
        sum += input[in_pos_idx];
    }

    // Block reduction
    __shared__ float partial_sums[BlockSize];
    partial_sums[tid] = sum;
    __syncthreads();

    for (int s = BlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = partial_sums[0] / static_cast<float>(reduction_size);
    }
}

void mean_reduction_cuda(
    torch::Tensor input,
    int dim,
    torch::Tensor& output
) {
    const int num_dims = input.dim();
    const int block_size = 256;
    const int total_out = output.numel();

    auto input_dims = input.sizes().vec();
    auto output_dims = output.sizes().vec();
    auto input_strides = input.strides().vec();

    // Convert to device tensors
    torch::Tensor input_dims_tensor = torch::tensor(input_dims, torch::kInt32).to(input.device());
    torch::Tensor output_dims_tensor = torch::tensor(output_dims, torch::kInt32).to(input.device());
    torch::Tensor input_strides_tensor = torch::tensor(input_strides, torch::kInt32).to(input.device());

    dim3 threads(block_size);
    dim3 blocks(total_out);

    mean_reduction_kernel<block_size><<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        input_dims_tensor.data_ptr<int>(),
        output_dims_tensor.data_ptr<int>(),
        input_strides_tensor.data_ptr<int>(),
        input.size(dim),
        num_dims,
        total_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
}
"""

mean_reduction_cuda = load_inline(
    name="mean_reduction_cuda",
    cpp_sources="",
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        output_shape = x.size()[:self.dim] + x.size()[self.dim+1:]
        output = torch.empty(
            output_shape,
            dtype=x.dtype,
            device=x.device
        )
        mean_reduction_cuda(x, self.dim, output)
        return output