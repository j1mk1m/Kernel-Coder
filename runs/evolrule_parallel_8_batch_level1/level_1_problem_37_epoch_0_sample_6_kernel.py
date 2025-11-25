import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_squares_kernel(const float* __restrict__ data, float* block_sums, int64_t size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_threads = blockDim.x;
    
    float sum = 0.0f;
    for (int i = bid * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
        float val = data[i];
        sum += val * val;
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_sums[bid] = shared[0];
    }
}

__global__ void reduce_block_sums(float* block_sums, int num_blocks, float* global_sum) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_sums[i];
    }
    
    shared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(global_sum, shared[0]);
    }
}

__global__ void divide_kernel(const float* input, float* output, float norm, int64_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        output[tid] = input[tid] / norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    const int64_t size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Allocate block_sums on device
    float* block_sums_dev;
    cudaMalloc(&block_sums_dev, num_blocks * sizeof(float));
    cudaMemset(block_sums_dev, 0, num_blocks * sizeof(float));

    // Launch sum_squares_kernel
    sum_squares_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), block_sums_dev, size);

    // Allocate global_sum on device
    float* global_sum_dev;
    cudaMalloc(&global_sum_dev, sizeof(float));
    cudaMemset(global_sum_dev, 0, sizeof(float));

    // Reduce block_sums to global_sum
    int reduce_block_size = 256;
    reduce_block_sums<<<1, reduce_block_size, reduce_block_size * sizeof(float)>>>(
        block_sums_dev, num_blocks, global_sum_dev);

    // Copy global_sum to host
    float global_sum_host;
    cudaMemcpy(&global_sum_host, global_sum_dev, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(block_sums_dev);
    cudaFree(global_sum_dev);

    float norm = sqrt(global_sum_host);

    // Create output tensor
    auto output = torch::empty_like(input);
    divide_kernel<<<(size + block_size - 1) / block_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), norm, size);

    return output;
}
"""

cpp_source = """
torch::Tensor frobenius_norm_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm.frobenius_norm_cuda(x)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []