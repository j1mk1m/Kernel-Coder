import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D average pooling
avg_pool_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void avg_pool3d_kernel(const scalar_t* input, scalar_t* output,
                                 int batch_size, int channels, int in_depth, int in_height, int in_width,
                                 int kernel_depth, int kernel_height, int kernel_width,
                                 int stride_depth, int stride_height, int stride_width,
                                 int padding_depth, int padding_height, int padding_width,
                                 int out_depth, int out_height, int out_width) {

    const int output_size = batch_size * channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    // Compute the output coordinates
    int batch = idx / (channels * out_depth * out_height * out_width);
    int remaining = idx % (channels * out_depth * out_height * out_width);
    int channel = remaining / (out_depth * out_height * out_width);
    remaining = remaining % (out_depth * out_height * out_width);
    int od = remaining / (out_height * out_width);
    remaining = remaining % (out_height * out_width);
    int oh = remaining / out_width;
    int ow = remaining % out_width;

    // Compute start positions in padded input
    int start_d = od * stride_depth - padding_depth;
    int start_h = oh * stride_height - padding_height;
    int start_w = ow * stride_width - padding_width;

    float sum = 0.0;
    int kernel_volume = kernel_depth * kernel_height * kernel_width;

    for (int d = 0; d < kernel_depth; ++d) {
        int padded_d = start_d + d;
        int original_d = padded_d - padding_depth;
        if (original_d < 0 || original_d >= in_depth) continue;

        for (int h = 0; h < kernel_height; ++h) {
            int padded_h = start_h + h;
            int original_h = padded_h - padding_height;
            if (original_h < 0 || original_h >= in_height) continue;

            for (int w = 0; w < kernel_width; ++w) {
                int padded_w = start_w + w;
                int original_w = padded_w - padding_width;
                if (original_w < 0 || original_w >= in_width) continue;

                // Compute input offset
                int in_offset = batch * channels * in_depth * in_height * in_width +
                                channel * in_depth * in_height * in_width +
                                original_d * in_height * in_width +
                                original_h * in_width +
                                original_w;

                sum += input[in_offset];
            }
        }
    }

    output[idx] = sum / kernel_volume;
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_depth, int kernel_height, int kernel_width,
                             int stride_depth, int stride_height, int stride_width,
                             int padding_depth, int padding_height, int padding_width) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_depth = input.size(2);
    const auto in_height = input.size(3);
    const auto in_width = input.size(4);

    // Calculate output dimensions
    auto out_depth = (in_depth + 2 * padding_depth - kernel_depth) / stride_depth + 1;
    auto out_height = (in_height + 2 * padding_height - kernel_height) / stride_height + 1;
    auto out_width = (in_width + 2 * padding_width - kernel_width) / stride_width + 1;

    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * channels * out_depth * out_height * out_width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool3d_cuda", ([&] {
        avg_pool3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, channels, in_depth, in_height, in_width,
            kernel_depth, kernel_height, kernel_width,
            stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width,
            out_depth, out_height, out_width
        );
    }));

    return output;
}
"""

avg_pool_3d_cpp_source = """
torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_depth, int kernel_height, int kernel_width,
                             int stride_depth, int stride_height, int stride_width,
                             int padding_depth, int padding_height, int padding_width);
"""

# Compile the inline CUDA code for 3D average pooling
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool_3d_cpp_source,
    cuda_sources=avg_pool_3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.avg_pool_cuda = avg_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        return self.avg_pool_cuda(x, kernel_d, kernel_h, kernel_w,
                                 stride_d, stride_h, stride_w,
                                 padding_d, padding_h, padding_w)