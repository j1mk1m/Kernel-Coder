import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool1d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int padding,
    const int stride) {

    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;
    const int output_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (output_pos >= output_length) return;

    const int input_offset = batch_idx * channels * input_length + channel_idx * input_length;
    const int output_offset = batch_idx * channels * output_length + channel_idx * output_length;

    scalar_t sum = 0.0;
    const int start = -padding + output_pos * stride;
    const int end = start + kernel_size;

    for (int i = start; i < end; ++i) {
        if (i >= 0 && i < input_length) {
            sum += input[input_offset + i];
        }
    }

    output[output_offset + output_pos] = sum / static_cast<scalar_t>(kernel_size);
}

torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);

    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    const int threads = 256;
    const dim3 blocks(batch_size, channels, (output_length + threads - 1) / threads);
    const dim3 threadsPerBlock(threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool1d_cuda", ([&] {
        avg_pool1d_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_length,
            output_length,
            kernel_size,
            padding,
            stride);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

avg_pool1d_cpp_source = (
    "torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for avg_pool1d
avg_pool1d = load_inline(
    name="avg_pool1d",
    cpp_sources=avg_pool1d_cpp_source,
    cuda_sources=avg_pool1d_source,
    functions=["avg_pool1d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool_cuda = avg_pool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool1d_cuda(x, self.kernel_size, self.stride, self.padding)

# Ensure the inputs are on the GPU
def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]