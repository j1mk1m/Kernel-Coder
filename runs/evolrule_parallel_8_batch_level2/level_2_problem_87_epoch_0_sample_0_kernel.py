import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for post-convolution processing (subtract and mish)
conv_post_process_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void conv_post_process_kernel(
    const float* input, float v1, float v2, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] - (v1 + v2);
        float sp = logf(1.0f + expf(temp));
        float tanh_sp = tanhf(sp);
        output[idx] = temp * tanh_sp;
    }
}

torch::Tensor conv_post_process_cuda(torch::Tensor input, float v1, float v2) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    // Launch the kernel
    conv_post_process_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), v1, v2, output.data_ptr<float>(), size);

    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(launch_err)));
    }

    // Synchronize to check for execution errors
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution failed: " + std::string(cudaGetErrorString(sync_err)));
    }

    return output;
}
"""

conv_post_process_cpp_source = """
#include <torch/extension.h>

torch::Tensor conv_post_process_cuda(torch::Tensor input, float v1, float v2);
"""

# Compile the inline CUDA code
conv_post_process = load_inline(
    name="conv_post_process",
    cpp_sources=conv_post_process_cpp_source,
    cuda_sources=conv_post_process_source,
    functions=["conv_post_process_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.conv_post_process = conv_post_process  # Custom CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_post_process.conv_post_process_cuda(
            x, self.subtract_value_1, self.subtract_value_2
        )
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]