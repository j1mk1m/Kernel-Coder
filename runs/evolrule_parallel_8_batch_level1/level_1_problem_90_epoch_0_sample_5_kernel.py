import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumprod_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cumprod_kernel(scalar_t* __restrict__ out,
                              const scalar_t* __restrict__ in,
                              int batch_size,
                              int dim_size,
                              int dim) {
    const int row_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int threads_per_block = blockDim.x;

    // Each thread processes a segment of the row
    const int segment_size = (dim_size + threads_per_block - 1) / threads_per_block;
    const int start = thread_idx * segment_size;
    const int end = min(start + segment_size, dim_size);

    // Load the segment into registers
    scalar_t local_data[segment_size];
    for (int i = 0; i < segment_size; ++i) {
        const int pos = start + i;
        if (pos < dim_size) {
            local_data[i] = in[row_idx * dim_size + pos];
        } else {
            local_data[i] = 1.0; // padding for beyond array
        }
    }

    // Compute cumulative product within the segment
    scalar_t local_cumprod[segment_size];
    scalar_t segment_product = 1.0;
    for (int i = 0; i < segment_size; ++i) {
        if (i == 0) {
            local_cumprod[i] = local_data[i];
        } else {
            local_cumprod[i] = local_cumprod[i-1] * local_data[i];
        }
        segment_product *= local_data[i];
    }

    // Store segment product in shared memory
    __shared__ scalar_t segment_products[threads_per_block];
    segment_products[thread_idx] = segment_product;
    __syncthreads();

    // Compute prefix product of segment products
    __shared__ scalar_t scan_results[threads_per_block];
    scan_results[thread_idx] = segment_products[thread_idx];
    __syncthreads();

    for (int d = 1; d < threads_per_block; d *= 2) {
        const int span = d * 2;
        const int index = threadIdx.x;
        if (index >= d && index < span) {
            scan_results[index] *= scan_results[index - d];
        }
        __syncthreads();
    }

    // Compute C_prev (prefix up to previous segment)
    const scalar_t C_prev = (thread_idx == 0) ? 1.0 : scan_results[thread_idx - 1];

    // Write results to output
    for (int i = 0; i < segment_size; ++i) {
        const int pos = start + i;
        if (pos < dim_size) {
            out[row_idx * dim_size + pos] = C_prev * local_cumprod[i];
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1); // since dim=1

    const int threads_per_block = 1024;
    dim3 blocks(batch_size);
    dim3 threads(threads_per_block);

    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            dim
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
"""

cumprod = load_inline(
    name="cumprod_cuda",
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return cumprod.cumprod_cuda(x, self.dim)

# Rest of the code remains as per original
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]