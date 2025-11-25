import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel source for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l2_norm_kernel(float* in_data, float* out_data, int64_t batch_size, int64_t dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    int tid = threadIdx.x;

    float partial_sum = 0.0f;

    // Compute partial sum of squares for each thread's chunk
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = in_data[row * dim + i];
        partial_sum += val * val;
    }

    // Store partial sum in shared memory
    shared[tid] = partial_sum;
    __syncthreads();

    // Reduction in shared memory to compute total squared norm
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float total_squared = shared[0];
    float norm;
    if (tid == 0) {
        norm = sqrtf(total_squared);
        shared[0] = norm;
    }
    __syncthreads();
    norm = shared[0];

    // Perform division by norm for each element
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = in_data[row * dim + i];
        out_data[row * dim + i] = val / norm;
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor in) {
    auto batch_size = in.size(0);
    auto dim = in.size(1);

    // Ensure input is contiguous for row-major access
    in = in.contiguous();
    auto out = torch::empty_like(in);

    const int threads_per_block = 1024;
    const int blocks = batch_size;

    // Calculate shared memory size (floats) and convert to bytes
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // Launch kernel with proper configuration
    l2_norm_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return out;
}
"""

l2_norm_cpp_source = (
    "extern \"C\" torch::Tensor l2_norm_cuda(torch::Tensor in);"
)

# Compile the inline CUDA code
l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2_norm

    def forward(self, x):
        return self.l2_norm.l2_norm_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []