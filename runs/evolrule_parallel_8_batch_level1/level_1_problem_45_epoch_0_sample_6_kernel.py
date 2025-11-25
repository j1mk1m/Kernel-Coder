import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D Average Pooling
avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 const int batch_size,
                                 const int channels,
                                 const int input_height,
                                 const int input_width,
                                 const int kernel_size,
                                 const int stride,
                                 const int padding) {
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) return;

    const int w = idx % output_width;
    const int h = (idx / output_width) % output_height;
    const int c = (idx / output_width / output_height) % channels;
    const int n = idx / (output_width * output_height * channels);

    // Compute the input coordinates
    const int input_h_start = h * stride - padding;
    const int input_w_start = w * stride - padding;
    const int input_h_end = input_h_start + kernel_size;
    const int input_w_end = input_w_start + kernel_size;

    scalar_t sum = 0.0;
    for (int ih = input_h_start; ih < input_h_end; ++ih) {
        for (int iw = input_w_start; iw < input_w_end; ++iw) {
            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int input_idx = n * channels * input_height * input_width +
                                     c * input_height * input_width +
                                     ih * input_width + iw;
                sum += input[input_idx];
            }
        }
    }
    const int output_idx = n * channels * output_height * output_width +
                          c * output_height * output_width +
                          h * output_width + w;
    output[output_idx] = sum / (kernel_size * kernel_size);
}

std::vector<int64_t> output_size(const torch::Tensor& input,
                                const int kernel_size,
                                const int stride,
                                const int padding) {
    const auto input_size = input.sizes();
    const int batch_size = input_size[0];
    const int channels = input_size[1];
    const int input_height = input_size[2];
    const int input_width = input_size[3];

    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    return {batch_size, channels, output_height, output_width};
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding) {
    auto output_dims = output_size(input, kernel_size, stride, padding);
    auto output = torch::empty(output_dims, input.options());

    const int threads = 256;
    const int elements = output.numel();
    const int blocks = (elements + threads - 1) / threads;

    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool2d_cuda", ([&] {
        avg_pool2d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            input.size(0),
            input.size(1),
            input.size(2),
            input.size(3),
            kernel_size,
            stride,
            padding);
    }));

    return output;
}
"""

avg_pool2d_cpp_source = """
std::vector<int64_t> output_size(const torch::Tensor& input,
                                const int kernel_size,
                                const int stride,
                                const int padding);
torch::Tensor avg_pool2d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding);
"""

# Compile the inline CUDA code for 2D Average Pooling
avg_pool2d = load_inline(
    name="avg_pool2d",
    cpp_sources=avg_pool2d_cpp_source,
    cuda_sources=avg_pool2d_source,
    functions=["avg_pool2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_cuda = avg_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool2d_cuda(
            x, self.kernel_size, self.stride, self.padding
        )

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11