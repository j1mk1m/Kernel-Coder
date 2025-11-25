import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

#define KERNEL_SIZE 8
#define STRIDE 1
#define PADDING 4
#define DILATION 3

__global__ void max_pool1d_kernel(
    const float* input, 
    float* output, 
    int batch_size, 
    int features, 
    int input_length, 
    int output_length) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features * output_length)
        return;

    int n = idx / (features * output_length);
    int rem = idx % (features * output_length);
    int c = rem / output_length;
    int o = rem % output_length;

    int start_padded = o * STRIDE;

    int p0 = start_padded + 0 * DILATION;
    int p1 = start_padded + 1 * DILATION;
    int p2 = start_padded + 2 * DILATION;
    int p3 = start_padded + 3 * DILATION;
    int p4 = start_padded + 4 * DILATION;
    int p5 = start_padded + 5 * DILATION;
    int p6 = start_padded + 6 * DILATION;
    int p7 = start_padded + 7 * DILATION;

    float max_val = -std::numeric_limits<float>::infinity();
    int input_length_padded = input_length + 2 * PADDING;

    #define PROCESS_P(p) \\
    if (p >= 0 && p < input_length_padded) { \\
        int original_p = p - PADDING; \\
        float val; \\
        if (original_p < 0 || original_p >= input_length) { \\
            val = 0.0f; \\
        } else { \\
            val = input[n * features * input_length + c * input_length + original_p]; \\
        } \\
        if (val > max_val) { \\
            max_val = val; \\
        } \\
    }

    PROCESS_P(p0);
    PROCESS_P(p1);
    PROCESS_P(p2);
    PROCESS_P(p3);
    PROCESS_P(p4);
    PROCESS_P(p5);
    PROCESS_P(p6);
    PROCESS_P(p7);

    #undef PROCESS_P

    int output_offset = n * features * output_length + c * output_length + o;
    output[output_offset] = max_val;
}

torch::Tensor max_pool1d_cuda(torch::Tensor input) {
    auto input_device = input.device();
    if (input_device.type() != torch::kCUDA) {
        throw std::runtime_error("Input must be on CUDA device");
    }

    const int batch_size = input.size(0);
    const int features = input.size(1);
    const int input_length = input.size(2);

    const int effective_kernel_size = (KERNEL_SIZE - 1) * DILATION + 1;
    const int output_length = (input_length + 2 * PADDING - effective_kernel_size) / STRIDE + 1;

    auto output = torch::zeros({batch_size, features, output_length}, input.options());

    const int total_threads = batch_size * features * output_length;
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    max_pool1d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        input_length,
        output_length
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

max_pool1d_cpp_source = (
    "torch::Tensor max_pool1d_cuda(torch::Tensor input);"
)

max_pool1d = load_inline(
    name="max_pool1d",
    cpp_sources=max_pool1d_cpp_source,
    cuda_sources=max_pool1d_source,
    functions=["max_pool1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool_cuda = max_pool1d

    def forward(self, x):
        return self.max_pool_cuda.max_pool1d_cuda(x)