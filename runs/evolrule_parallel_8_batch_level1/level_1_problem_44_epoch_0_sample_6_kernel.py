import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 1D Average Pooling
avg_pool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void avg_pool1d_forward_kernel(const scalar_t* __restrict__ input,
                                         scalar_t* __restrict__ output,
                                         int64_t batch_size,
                                         int64_t channels,
                                         int64_t input_length,
                                         int64_t kernel_size,
                                         int64_t stride,
                                         int64_t padding) {
    const int64_t output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    CUDA_KERNEL_LOOP(index, batch_size * channels * output_length) {
        const int64_t n = index / (channels * output_length);
        const int64_t c = (index / output_length) % channels;
        const int64_t o = index % output_length;

        const int64_t input_start = o * stride - padding;
        const int64_t input_end = input_start + kernel_size;

        scalar_t sum = 0.0;
        for (int64_t i = input_start; i < input_end; ++i) {
            if (i >= 0 && i < input_length) {
                sum += input[n * channels * input_length + c * input_length + i];
            }
        }
        output[index] = sum / kernel_size;
    }
}

torch::Tensor avg_pool1d_forward_cuda(torch::Tensor input,
                                     int64_t kernel_size,
                                     int64_t stride,
                                     int64_t padding) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_length = input.size(2);

    const int64_t output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    const int threads = 256;
    const dim3 blocks((batch_size * channels * output_length + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool1d_forward", ([&] {
        avg_pool1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_length,
            kernel_size,
            stride,
            padding
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

avg_pool_1d_cpp_source = """
torch::Tensor avg_pool1d_forward_cuda(torch::Tensor input,
                                     int64_t kernel_size,
                                     int64_t stride,
                                     int64_t padding);
"""

# Compile the CUDA kernel
avg_pool_1d = load_inline(
    name="avg_pool_1d",
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_source,
    functions=["avg_pool1d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool_cuda = avg_pool_1d  # Reference to the CUDA module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is on the correct device
        x = x.contiguous()
        return self.avg_pool_cuda.avg_pool1d_forward_cuda(x, self.kernel_size, self.stride, self.padding)

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]

# Constants from original problem
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4