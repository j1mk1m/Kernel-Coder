import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for parallel exclusive prefix sum (scan)
scan_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void exclusive_scan_2d_kernel(const T* input, T* output, int batch_size, int dim_size, int dim) {
    // Each block handles a single batch element along the batch dimension
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // Shared memory for block processing
    extern __shared__ T sdata[];

    // Load the current row into shared memory
    int base_offset = batch_idx * dim_size;
    T val = input[base_offset + thread_idx];
    sdata[thread_idx] = val;
    __syncthreads();

    // Perform parallel scan within the block
    for (int stride = 1; stride <= thread_idx; stride *= 2) {
        T temp = 0;
        if (stride <= thread_idx) {
            temp = sdata[thread_idx - stride];
        }
        __syncthreads();
        sdata[thread_idx] += temp;
        __syncthreads();
    }

    // Write the result back to global memory (exclusive scan: subtract current value)
    output[base_offset + thread_idx] = (thread_idx == 0) ? 0 : sdata[thread_idx] - val;
}

torch::Tensor exclusive_scan_2d_cuda(torch::Tensor input, int dim) {
    const int batch_size = input.size(0);
    const int dim_size = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;  // Must be >= maximum dimension size
    dim3 grid(batch_size);
    dim3 block(dim_size);  // Each thread handles one element in the dimension

    // Ensure block size is sufficient for the dimension size
    if (dim_size > block_size) {
        // Handle larger dimensions by using multiple threads per element (not implemented here)
        // For simplicity, assume dim_size <= block_size for now
        AT_ASSERT(dim_size <= block_size);
    }

    // Allocate shared memory for the block
    size_t shared_mem = dim_size * sizeof(float);

    // Launch kernel for exclusive scan along dim=1 (columns)
    exclusive_scan_2d_kernel<float><<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size,
        dim
    );

    cudaDeviceSynchronize();
    return output;
}
"""

scan_cuda_header = """
torch::Tensor exclusive_scan_2d_cuda(torch::Tensor input, int dim);
"""

# Compile the CUDA kernel
scan_cuda = load_inline(
    name="scan_cuda",
    cpp_sources=scan_cuda_header,
    cuda_sources=scan_cuda_source,
    functions=["exclusive_scan_2d_cuda"],
    verbose=True,
    extra_cflags=["-D_ENABLE_CUDA_FUSED"],
    extra_cuda_cflags=["-lineinfo"]
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.scan_cuda = scan_cuda  # Store the loaded module

    def forward(self, x):
        return self.scan_cuda.exclusive_scan_2d_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]