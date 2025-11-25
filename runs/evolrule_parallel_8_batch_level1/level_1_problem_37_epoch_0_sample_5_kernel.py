import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm normalization
frob_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void frob_norm_kernel(
    float* input,
    float* output,
    int size,
    float* partial_sums,
    int num_blocks,
    int* counter,
    float* total_s,
    bool* done
) {
    __shared__ float shared_squares[256];  // Shared memory for block reduction

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    float local_sum = 0.0f;

    // Phase 1: Compute partial sum for this block using grid-stride
    while (idx < size) {
        float val = input[idx];
        local_sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    // Store local sum to shared memory
    shared_squares[tid] = local_sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_squares[tid] += shared_squares[tid + s];
        }
        __syncthreads();
    }

    // Write block's partial sum to global array and update counter
    if (tid == 0) {
        partial_sums[bid] = shared_squares[0];
        atomicAdd(counter, 1);
    }
    __syncthreads();

    // Phase 2: Compute total norm if this is the first thread of the first block
    if (bid == 0 && tid == 0) {
        // Spin until all blocks have written their partial sums
        while (atomicLoad(counter) < num_blocks) {}

        // Compute total sum from partial_sums
        float total = 0.0f;
        for (int i = 0; i < num_blocks; ++i) {
            total += partial_sums[i];
        }
        *total_s = total;
        *done = true;
    }
    __syncthreads();

    // Phase 3: Wait until total is computed and normalize
    while (!*done) {}

    // Compute norm and handle zero case
    float norm = sqrt(*total_s);
    if (norm == 0.0f) norm = 1.0f;  // Avoid division by zero

    // Normalize elements using grid-stride
    idx = bid * blockDim.x + tid;
    while (idx < size) {
        output[idx] = input[idx] / norm;
        idx += blockDim.x * gridDim.x;
    }
}

torch::Tensor frob_norm_cuda(torch::Tensor input) {
    auto size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Allocate output and auxiliary tensors
    auto output = torch::empty_like(input);
    auto partial_sums = torch::empty({num_blocks}, torch::kFloat32, torch::device("cuda"));
    auto counter = torch::zeros({1}, torch::kInt32, torch::device("cuda"));
    auto total_s = torch::empty({1}, torch::kFloat32, torch::device("cuda"));
    auto done = torch::empty({1}, torch::kBool, torch::device("cuda"));

    // Launch kernel
    frob_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        partial_sums.data_ptr<float>(),
        num_blocks,
        counter.data_ptr<int>(),
        total_s.data_ptr<float>(),
        done.data_ptr<bool>()
    );

    // Synchronize to ensure completion
    cudaDeviceSynchronize();

    return output;
}
"""

cpp_source = """
torch::Tensor frob_norm_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
frob_norm = load_inline(
    name="frob_norm",
    cpp_sources=cpp_source,
    cuda_sources=frob_norm_source,
    functions=["frob_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.frob_norm = frob_norm

    def forward(self, x):
        return self.frob_norm.frob_norm_cuda(x)