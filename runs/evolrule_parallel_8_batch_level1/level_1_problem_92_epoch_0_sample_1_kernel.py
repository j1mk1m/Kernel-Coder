import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void exclusive_cumsum_kernel(
    const scalar_t* input,
    scalar_t* output,
    int dim_size,
    int num_vectors,
    int input_stride,
    int output_stride
) {
    int vector_id = blockIdx.x;
    if (vector_id >= num_vectors) return;

    const scalar_t* input_row = input + vector_id * input_stride;
    scalar_t* output_row = output + vector_id * output_stride;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    __shared__ scalar_t s_data[1024 * 2]; // Double the block size for padding

    // Load data into shared memory
    for (int i = tid; i < dim_size; i += block_size) {
        s_data[i] = input_row[i];
    }
    __syncthreads();

    // Up-sweep phase
    for (int s = 1; s < dim_size; s *= 2) {
        int index = 2 * s * tid;
        if (index < dim_size) {
            s_data[index + s] += s_data[index];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int s = dim_size / 2; s > 0; s /= 2) {
        int index = 2 * s * tid;
        if (index < dim_size) {
            scalar_t temp = s_data[index];
            s_data[index] = s_data[index + s];
            s_data[index + s] += temp;
        }
        __syncthreads();
    }

    // Shift left by one to get exclusive sum
    if (tid < dim_size) {
        if (tid == 0) {
            output_row[tid] = 0;
        } else {
            output_row[tid] = s_data[tid - 1];
        }
    }
}

void exclusive_cumsum_cuda(torch::Tensor input, torch::Tensor output, int dim_size, int num_vectors, int input_stride, int output_stride) {
    dim3 blocks(num_vectors);
    dim3 threads(1024);
    int shared_size = (dim_size < 1024 * 2 ? dim_size : 1024 * 2) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "exclusive_cumsum_cuda", ([&] {
        exclusive_cumsum_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            num_vectors,
            input_stride,
            output_stride
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
}
"""

exclusive_cumsum_cpp = """
#include <torch/extension.h>

void exclusive_cumsum_cuda(torch::Tensor input, torch::Tensor output, int dim_size, int num_vectors, int input_stride, int output_stride);
"""

exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dim_size = x.size(self.dim)
        num_vectors = x.size(0) if self.dim == 1 else x.size(1)
        input_stride = x.stride(self.dim)
        output_stride = x.stride(self.dim)

        output = torch.empty_like(x)
        exclusive_cumsum(x, output, dim_size, num_vectors, input_stride, output_stride)
        return output