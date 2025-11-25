import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D Average Pooling
avg_pool_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool3d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding) {

    const int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    // Compute output element indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_depth * out_height * out_width) return;

    // Calculate the output's indices
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int d_out = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % channels;
    int n = idx / (out_width * out_height * out_depth * channels);

    // Compute input indices with padding
    int in_d_start = d_out * stride - padding;
    int in_h_start = h_out * stride - padding;
    int in_w_start = w_out * stride - padding;

    scalar_t sum = 0.0;
    int valid_count = 0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        int in_d = in_d_start + kd;
        if (in_d < 0 || in_d >= in_depth) continue;

        for (int kh = 0; kh < kernel_size; ++kh) {
            int in_h = in_h_start + kh;
            if (in_h < 0 || in_h >= in_height) continue;

            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_w = in_w_start + kw;
                if (in_w < 0 || in_w >= in_width) continue;

                // Compute input index
                int in_idx = n * channels * in_depth * in_height * in_width +
                            c * in_depth * in_height * in_width +
                            in_d * in_height * in_width +
                            in_h * in_width +
                            in_w;

                sum += input[in_idx];
                valid_count++;
            }
        }
    }

    // Compute average
    output[idx] = sum / valid_count;
}

torch::Tensor avg_pool3d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_depth = (in_depth + 2 * padding - kernel_size) / stride + 1;
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, input.options());

    const int num_threads = 512;
    const int num_elements = batch_size * channels * out_depth * out_height * out_width;
    const int num_blocks = (num_elements + num_threads - 1) / num_threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool3d_forward_cuda", ([&] {
        avg_pool3d_forward_kernel<scalar_t><<<num_blocks, num_threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            in_depth,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding);
    }));

    return output;
}
"""

avg_pool_3d_cpp_source = """
torch::Tensor avg_pool3d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding);
"""

# Compile the CUDA kernel
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool_3d_cpp_source,
    cuda_sources=avg_pool_3d_source,
    functions=["avg_pool3d_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool_cuda = avg_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool3d_forward_cuda(
            x, self.kernel_size, self.stride, self.padding
        )