import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class FusedConvTransposeScaleBias(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, bias_shape):
        super(FusedConvTransposeScaleBias, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Apply convolution transpose and scaling in a fused kernel
        # Also add bias in the same kernel
        # Custom kernel implementation here
        return fused_conv_transpose_scale_add(x, self.conv_transpose.weight, self.conv_transpose.bias, 
                                             self.scale, self.bias, self.conv_transpose.stride,
                                             self.conv_transpose.padding, self.conv_transpose.output_padding)

# Define the fused ConvTranspose + Scale + Add kernel
fused_conv_transpose_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void fused_conv_transpose_scale_add_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> bias,
    const scalar_t scale,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_size, int stride, int padding,
    int output_padding, int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width) {

    CUDA_1D_KERNEL_LOOP(index, output_depth * output_height * output_width * out_channels) {
        int output_c = index / (output_depth * output_height * output_width);
        int rest = index % (output_depth * output_height * output_width);
        int d = rest / (output_height * output_width);
        int h = (rest % (output_height * output_width)) / output_width;
        int w = rest % output_width;

        scalar_t sum = 0;
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int input_d = (d - kd) / stride;
                        int input_h = (h - kh) / stride;
                        int input_w = (w - kw) / stride;

                        // Check if within input bounds
                        if (input_d < 0 || input_d >= input_depth ||
                            input_h < 0 || input_h >= input_height ||
                            input_w < 0 || input_w >= input_width) {
                            continue;
                        }

                        sum += weight[in_c][output_c][kd][kh][kw] * 
                               input[input_d][in_c][input_h][input_w];
                    }
                }
            }
        }
        sum = sum * scale + bias[output_c];
        output[output_c][d][h][w] = sum;
    }
}

std::vector<torch::Tensor> fused_conv_transpose_scale_add(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale,
    std::tuple<int, int, int> stride,
    std::tuple<int, int, int> padding,
    std::tuple<int, int, int> output_padding) {

    const int in_channels = weight.size(0);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2); // Assuming cubic kernel
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    // Calculate output dimensions
    const int output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_size + output_padding[0];
    const int output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_size + output_padding[1];
    const int output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_size + output_padding[2];

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({input.size(0), out_channels, output_depth, output_height, output_width}, output_options);

    dim3 blocks = 128;
    dim3 threads = 1024;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_transpose_scale_add_cuda", ([&] {
        fused_conv_transpose_scale_add_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
            scale,
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            in_channels, out_channels, kernel_size, stride[0], padding[0], output_padding[0],
            input_depth, input_height, input_width,
            output_depth, output_height, output_width);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_scale_add", &fused_conv_transpose_scale_add, "Fused ConvTranspose + Scale + Add");
}
"""

# Compile the fused kernel
fused_conv_transpose = load_inline(
    name="fused_conv_transpose",
    cuda_sources=fused_conv_transpose_source,
    functions=["fused_conv_transpose_scale_add"],
    verbose=True
)

class FusedAvgPoolScale(nn.Module):
    def __init__(self, kernel_size, scale):
        super(FusedAvgPoolScale, self).__init__()
        self.kernel_size = kernel_size
        self.scale = scale

    def forward(self, x):
        return fused_avg_pool_scale(x, self.kernel_size, self.scale)

avg_pool_scale_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_avg_pool_scale_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int kernel_size, scalar_t scale,
    int in_channels, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width) {

    CUDA_1D_KERNEL_LOOP(index, out_depth * out_height * out_width * in_channels) {
        int c = index / (out_depth * out_height * out_width);
        int d = (index % (out_depth * out_height * out_width)) / (out_height * out_width);
        int h = (index % (out_height * out_width)) / out_width;
        int w = index % out_width;

        scalar_t sum = 0;
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_d = d * kernel_size + kd;
                    int input_h = h * kernel_size + kh;
                    int input_w = w * kernel_size + kw;

                    if (input_d < in_depth && input_h < in_height && input_w < in_width) {
                        sum += input[c][input_d][input_h][input_w];
                    }
                }
            }
        }
        output[c][d][h][w] = (sum / (kernel_size*kernel_size*kernel_size)) * scale;
    }
}

std::vector<torch::Tensor> fused_avg_pool_scale(
    torch::Tensor input,
    int kernel_size,
    float scale) {

    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_depth = in_depth / kernel_size;
    const int out_height = in_height / kernel_size;
    const int out_width = in_width / kernel_size;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({input.size(0), in_channels, out_depth, out_height, out_width}, output_options);

    dim3 blocks = 128;
    dim3 threads = 1024;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_avg_pool_scale_cuda", ([&] {
        fused_avg_pool_scale_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            kernel_size, scale,
            in_channels, in_depth, in_height, in_width,
            out_depth, out_height, out_width);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_avg_pool_scale", &fused_avg_pool_scale, "Fused AvgPool + Scale");
}
"""

# Compile the fused average pool + scale kernel
fused_avg_pool = load_inline(
    name="fused_avg_pool",
    cuda_sources=avg_pool_scale_source,
    functions=["fused_avg_pool_scale"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.fused_conv_transpose = FusedConvTransposeScaleBias(
            in_channels, out_channels, kernel_size, stride, padding, scale1, bias_shape)
        self.avg_pool_scale = FusedAvgPoolScale(2, scale2)  # Assuming kernel_size=2 for AvgPool

    def forward(self, x):
        x = self.fused_conv_transpose(x)
        x = self.avg_pool_scale(x)
        return x

# Ensure compatibility with original parameters
ModelNew.forward = ModelNew.forward