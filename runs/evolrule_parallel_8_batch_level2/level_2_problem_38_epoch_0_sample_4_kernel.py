import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for AvgPool3d
avg_pool3d_cuda = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void avg_pool3d_kernel(
    const T* input, T* output,
    int batch, int channels, int in_d, int in_h, int in_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels) return;

    int channel = idx % channels;
    int b = idx / channels;

    for (int oz = 0; oz < (in_d - kernel_d)/stride_d + 1; oz++) {
        for (int oh = 0; oh < (in_h - kernel_h)/stride_h + 1; oh++) {
            for (int ow = 0; ow < (in_w - kernel_w)/stride_w + 1; ow++) {
                int iz_start = oz * stride_d;
                int ih_start = oh * stride_h;
                int iw_start = ow * stride_w;
                T sum = 0;
                for (int kz = 0; kz < kernel_d; kz++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int iz = iz_start + kz;
                            int ih = ih_start + kh;
                            int iw = iw_start + kw;
                            int input_offset = b * channels * in_d * in_h * in_w +
                                channel * in_d * in_h * in_w +
                                iz * in_h * in_w +
                                ih * in_w +
                                iw;
                            sum += input[input_offset];
                        }
                    }
                }
                int output_offset = b * channels * ((in_d - kernel_d)/stride_d + 1) *
                    ((in_h - kernel_h)/stride_h + 1) * ((in_w - kernel_w)/stride_w + 1) +
                    channel * ((in_d - kernel_d)/stride_d + 1) * ((in_h - kernel_h)/stride_h + 1) * ((in_w - kernel_w)/stride_w + 1) +
                    oz * ((in_h - kernel_h)/stride_h + 1) * ((in_w - kernel_w)/stride_w + 1) +
                    oh * ((in_w - kernel_w)/stride_w + 1) +
                    ow;
                output[output_offset] = sum / (kernel_d * kernel_h * kernel_w);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> avg_pool3d_cuda_forward(
    torch::Tensor input,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w) {

    const auto batch = input.size(0);
    const auto channels = input.size(1);
    const auto in_d = input.size(2);
    const auto in_h = input.size(3);
    const auto in_w = input.size(4);

    const int out_d = (in_d - kernel_d) / stride_d + 1;
    const int out_h = (in_h - kernel_h) / stride_h + 1;
    const int out_w = (in_w - kernel_w) / stride_w + 1;

    auto output = torch::zeros({batch, channels, out_d, out_h, out_w}, input.options());
    const int threads = 256;
    const int blocks = (batch * channels + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool3d_forward", ([&] {
        avg_pool3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch, channels, in_d, in_h, in_w,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w);
    }));

    return std::make_tuple(output, input); // Save input for backward
}

torch::Tensor avg_pool3d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w) {

    const auto batch = grad_output.size(0);
    const auto channels = grad_output.size(1);
    const auto out_d = grad_output.size(2);
    const auto out_h = grad_output.size(3);
    const auto out_w = grad_output.size(4);

    const auto in_d = (out_d - 1) * stride_d + kernel_d;
    const auto in_h = (out_h - 1) * stride_h + kernel_h;
    const auto in_w = (out_w - 1) * stride_w + kernel_w;

    auto grad_input = torch::zeros({batch, channels, in_d, in_h, in_w}, input.options());
    const int threads = 256;
    const int blocks = (batch * channels + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "avg_pool3d_backward", ([&] {
        avg_pool3d_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            batch, channels, in_d, in_h, in_w,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            out_d, out_h, out_w);
    }));

    return grad_input;
}
"""

# Define custom CUDA kernel for ConvTranspose3d
conv_transpose3d_cuda = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Implementation would be here but omitted for brevity. A full implementation
// would require handling padding, output padding, strides, and weights. Due to
// complexity and length, this is left as a placeholder with a note.

// [Note: Actual implementation requires significant kernel code for convolution,
// including handling weights, padding, strides, and output padding. This example
// demonstrates the pattern but the full kernel is omitted here for brevity.]

"""

# Define custom clamp kernel
clamp_cuda = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void clamp_kernel(const T* input, T* output, T min_val, T max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T val = input[idx];
        output[idx] = val < min_val ? min_val : (val > max_val ? max_val : val);
    }
}

torch::Tensor clamp_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "clamp_cuda", ([&] {
        clamp_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            min_val,
            max_val,
            size);
    }));

    return output;
}
"""

# Define spatial softmax kernel
spatial_softmax_cuda = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void spatial_softmax_kernel(
    const T* input, T* output, int batch, int channels, int spatial_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels) return;

    int channel = idx % channels;
    int b = idx / channels;

    T max_val = -INFINITY;
    for (int s = 0; s < spatial_size; s++) {
        T val = input[b * channels * spatial_size + channel * spatial_size + s];
        if (val > max_val) max_val = val;
    }

    T sum_exp = 0;
    for (int s = 0; s < spatial_size; s++) {
        T val = input[b * channels * spatial_size + channel * spatial_size + s];
        sum_exp += exp(val - max_val);
    }

    for (int s = 0; s < spatial_size; s++) {
        T val = input[b * channels * spatial_size + channel * spatial_size + s];
        output[b * channels * spatial_size + channel * spatial_size + s] 
            = exp(val - max_val) / sum_exp;
    }
}

torch::Tensor spatial_softmax_cuda(torch::Tensor input) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto spatial_size = input.size(2); // After view, spatial dims are flattened
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (batch * channels + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spatial_softmax", ([&] {
        spatial_softmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch, channels, spatial_size);
    }));

    return output;
}
"""

# Compile CUDA kernels
avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources="",
    cuda_sources=avg_pool3d_cuda,
    functions=["avg_pool3d_cuda_forward", "avg_pool3d_cuda_backward"],
    verbose=True
)

clamp_op = load_inline(
    name="clamp",
    cuda_sources=clamp_cuda,
    functions=["clamp_cuda"],
    verbose=True
)

spatial_softmax = load_inline(
    name="spatial_softmax",
    cuda_sources=spatial_softmax_cuda,
    functions=["spatial_softmax_cuda"],
    verbose=True
)

# ModelNew implementation
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool_kernel_d = pool_kernel_size
        self.avg_pool_kernel_h = pool_kernel_size
        self.avg_pool_kernel_w = pool_kernel_size
        self.avg_pool_stride_d = 1  # Assuming stride same as kernel for simplicity
        self.avg_pool_stride_h = 1
        self.avg_pool_stride_w = 1

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding
        )

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        # Custom AvgPool3d
        x, _ = avg_pool3d.avg_pool3d_cuda_forward(
            x,
            self.avg_pool_kernel_d, self.avg_pool_kernel_h, self.avg_pool_kernel_w,
            self.avg_pool_stride_d, self.avg_pool_stride_h, self.avg_pool_stride_w
        )

        # ConvTranspose3d remains as PyTorch's implementation due to complexity
        x = self.conv_transpose(x)

        # Custom Clamp
        x = clamp_op.clamp_cuda(x, self.clamp_min, self.clamp_max)

        # Spatial Softmax
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1)
        x = spatial_softmax.spatial_softmax_cuda(x)
        x = x.view(orig_shape)

        # Element-wise multiplication (basic operation, no need for custom kernel)
        x = x * self.scale

        return x