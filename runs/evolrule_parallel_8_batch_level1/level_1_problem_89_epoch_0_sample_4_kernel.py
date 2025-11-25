import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <cstdint>
#include <cstdio>

template <typename scalar_t>
__global__ void exclusive_scan(scalar_t *g_idata, scalar_t *g_odata, int n) {
    extern __shared__ scalar_t s_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    scalar_t carry = 0.0;

    // Load data into shared memory
    if (i < n) {
        s_data[tid] = g_idata[i];
        s_data[tid + blockDim.x] = g_idata[i + blockDim.x];
    } else {
        s_data[tid] = 0.0;
        s_data[tid + blockDim.x] = 0.0;
    }
    __syncthreads();

    // Up-sweep phase
    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        if (tid >= d) {
            s_data[tid] += s_data[tid - d];
        }
        __syncthreads();
    }

    // Down-sweep phase
    if (tid == 0) {
        s_data[blockDim.x * 2 - 1] = 0.0;
    }
    __syncthreads();
    for (int d = blockDim.x; d >= 1; d >>= 1) {
        if (tid >= d) {
            scalar_t temp = s_data[tid - d];
            __threadfence_block();
            s_data[tid] -= temp;
            s_data[tid + d] += temp;
        }
        __syncthreads();
    }

    // Write results to global memory
    if (i < n) {
        g_odata[i] = s_data[tid];
        g_odata[i + blockDim.x] = s_data[tid + blockDim.x];
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void inclusive_scan(scalar_t *g_idata, scalar_t *g_odata, int n) {
    __shared__ scalar_t s_data[512];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    scalar_t carry = 0.0;

    // Load data into shared memory
    if (i < n) {
        s_data[tid] = g_idata[i];
        s_data[tid + blockDim.x] = g_idata[i + blockDim.x];
    } else {
        s_data[tid] = 0.0;
        s_data[tid + blockDim.x] = 0.0;
    }
    __syncthreads();

    // Up-sweep phase
    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        if (tid >= d) {
            s_data[tid] += s_data[tid - d];
        }
        __syncthreads();
    }

    // Down-sweep phase with carry
    if (tid == 0) {
        s_data[blockDim.x * 2 - 1] = s_data[blockDim.x * 2 - 1] + carry;
        carry = s_data[blockDim.x * 2 - 1];
        s_data[blockDim.x * 2 - 1] = 0.0;
    }
    __syncthreads();
    for (int d = blockDim.x; d >= 1; d >>= 1) {
        if (tid >= d) {
            scalar_t temp = s_data[tid - d];
            __threadfence_block();
            s_data[tid] += temp;
            s_data[tid + d] -= temp;
        }
        __syncthreads();
    }

    // Write results to global memory
    if (i < n) {
        g_odata[i] = s_data[tid];
        g_odata[i + blockDim.x] = s_data[tid + blockDim.x];
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void block_sums(scalar_t *g_idata, scalar_t *block_sums, int n, int block_size) {
    __shared__ scalar_t s_data[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * block_size * 2 + threadIdx.x;
    scalar_t sum = 0.0;

    // Load data into shared memory
    if (i < n) {
        sum += g_idata[i];
        if (i + block_size < n) sum += g_idata[i + block_size];
    }
    s_data[tid] = sum;
    __syncthreads();

    // Reduce to block sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = s_data[0];
}

template <typename scalar_t>
void cuda_cumsum(torch::Tensor input, torch::Tensor output, int dim) {
    const int batch_size = input.size(0);
    const int length = input.size(1);
    const int block_size = 256;
    const dim3 threads(block_size);
    dim3 blocks((length + block_size - 1) / block_size);

    // Allocate temporary storage for block sums
    auto block_sums = torch::empty({batch_size}, input.options()).cuda();

    // Compute block sums for each batch and dimension
    block_sums.launch_kernel<block_sums<scalar_t>>(
        dim3(batch_size), threads,
        0, nullptr,
        input.data_ptr<scalar_t>(),
        block_sums.data_ptr<scalar_t>(),
        length,
        block_size
    );

    // Compute global prefix sum of block sums
    auto global_prefix = torch::empty_like(block_sums);
    inclusive_scan<scalar_t><<<1, block_size, block_size * sizeof(scalar_t)>>>(
        block_sums.data_ptr<scalar_t>(),
        global_prefix.data_ptr<scalar_t>(),
        batch_size
    );

    // Perform local scan on each block and adjust with global prefix
    for (int b = 0; b < batch_size; ++b) {
        auto input_b = input[b];
        auto output_b = output[b];
        scalar_t *input_ptr = input_b.data_ptr<scalar_t>();
        scalar_t *output_ptr = output_b.data_ptr<scalar_t>();

        inclusive_scan<scalar_t><<<blocks, threads, block_size * sizeof(scalar_t)>>>(
            input_ptr,
            output_ptr,
            length
        );

        // Apply global prefix to each block's scan result
        scalar_t prefix = (b == 0) ? 0 : global_prefix[b - 1].item<scalar_t>();
        for (int i = 0; i < length; ++i) {
            output_ptr[i] += prefix;
        }
    }
}

// Entry function for PyTorch
template <typename scalar_t>
torch::Tensor custom_cumsum(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    cuda_cumsum<scalar_t>(input, output, dim);
    return output;
}

// Specialize for float
torch::Tensor custom_cumsum_float(torch::Tensor input, int64_t dim) {
    return custom_cumsum<float>(input, dim);
}

// Specialize for double
torch::Tensor custom_cumsum_double(torch::Tensor input, int64_t dim) {
    return custom_cumsum<double>(input, dim);
}
"""

CUDA_HEADER = """
#include <torch/extension.h>
#include <cuda.h>

torch::Tensor custom_cumsum_float(torch::Tensor input, int64_t dim);
torch::Tensor custom_cumsum_double(torch::Tensor input, int64_t dim);
"""

# Compile the CUDA code inline
custom_cumsum = load_inline(
    name="custom_cumsum",
    cpp_sources=CUDA_HEADER,
    cuda_sources=CUDA_SOURCE,
    functions=[
        "custom_cumsum_float",
        "custom_cumsum_double"
    ],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.custom_cumsum = custom_cumsum

    def forward(self, x):
        return self.custom_cumsum.custom_cumsum_float(x, self.dim)