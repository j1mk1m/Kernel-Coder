import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_h,
    int out_w,
    int kernel_area
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_h * out_w) return;

    int w_out = idx % out_w;
    int rem = idx / out_w;
    int h_out = rem % out_h;
    int c = rem / out_h;
    int n = idx / (channels * out_h * out_w);

    scalar_t sum = 0.0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_out * stride_h + kh - pad_h;
            int w_in = w_out * stride_w + kw - pad_w;
            if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                int input_offset = n * channels * input_h * input_w +
                                   c * input_h * input_w +
                                   h_in * input_w + w_in;
                sum += input[input_offset];
            }
        }
    }

    int output_offset = n * channels * out_h * out_w +
                        c * out_h * out_w +
                        h_out * out_w + w_out;
    output[output_offset] = sum / kernel_area;
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input,
                             int kernel_h,
                             int kernel_w,
                             int stride_h,
                             int stride_w,
                             int pad_h,
                             int pad_w) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);

    // Compute output dimensions
    const int out_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    auto output = torch::empty({batch_size, channels, out_h, out_w}, input.options());

    const int num_elements = batch_size * channels * out_h * out_w;
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    int kernel_area_val = kernel_h * kernel_w;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool2d_cuda", ([&]{
        avg_pool2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            kernel_area_val
        );
    }));

    return output;
}
"""

avg_pool_cpp_source = (
    "torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);"
)

avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cpp_sources=avg_pool_cpp_source,
    cuda_sources=avg_pool_source,
    functions=["avg_pool2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.kernel_h = kernel_size
        self.kernel_w = kernel_size
        self.stride_h = self.stride
        self.stride_w = self.stride
        self.pad_h = padding
        self.pad_w = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool_cuda(x, self.kernel_h, self.kernel_w, self.stride_h, self.stride_w, self.pad_h, self.pad_w)