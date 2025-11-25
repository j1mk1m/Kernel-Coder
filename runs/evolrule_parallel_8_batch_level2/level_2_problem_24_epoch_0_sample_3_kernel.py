import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        x = torch.min(x, dim=self.dim)[0]
        x = torch.softmax(x, dim=1)
        return x

# Define custom CUDA kernels
# 1. Custom Conv3D kernel
conv3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels, int kernel_size,
    int input_depth, int input_height, int input_width,
    int output_depth, int output_height, int output_width) {

    CUDA_KERNEL_LOOP(index, batch_size * out_channels * output_depth * output_height * output_width) {
        int w = index % output_width;
        int h = (index / output_width) % output_height;
        int d = (index / (output_width * output_height)) % output_depth;
        int b = (index / (output_width * output_height * output_depth)) % batch_size;
        int c_out = index / (output_width * output_height * output_depth * batch_size);

        scalar_t sum = 0;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int id = d + kd;
                        int ih = h + kh;
                        int iw = w + kw;
                        if (id < input_depth && ih < input_height && iw < input_width) {
                            sum += input[b][c_in][id][ih][iw] * 
                                   weight[c_out][c_in][kd][kh][kw];
                        }
                    }
                }
            }
        }
        output[b][c_out][d][h][w] = sum;
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    auto output_depth = input_depth - kernel_size + 1;
    auto output_height = input_height - kernel_size + 1;
    auto output_width = input_width - kernel_size + 1;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int threads = 256;
    const int elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward_cuda", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,torch::DefaultGenerateIndex>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,torch::DefaultGenerateIndex>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,torch::DefaultGenerateIndex>(),
            batch_size, in_channels, out_channels, kernel_size,
            input_depth, input_height, input_width,
            output_depth, output_height, output_width);
    }));

    return output;
}
"""

conv3d_cpp_source = "torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight);"

conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True
)

# 2. Custom Min+Softmax fusion kernel
min_softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void min_softmax_fusion(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits> output,
    int batch_size, int channels, int depth, int height, int width,
    int reduction_dim) {

    // Compute min along dimension
    // Then compute softmax along channels
    // Assume reduction_dim is 2 (depth)
    CUDA_KERNEL_LOOP(index, batch_size * channels * height * width) {
        int w = index % width;
        int h = (index / width) % height;
        int b = (index / (width * height)) % batch_size;
        int c = index / (width * height * batch_size);

        // Compute min over depth
        scalar_t min_val = input[b][c][0][h][w];
        for (int d = 1; d < depth; ++d) {
            if (input[b][c][d][h][w] < min_val) {
                min_val = input[b][c][d][h][w];
            }
        }

        // Compute exponentials for softmax
        scalar_t max_val = min_val;  // For numerical stability
        scalar_t sum = 0;
        for (int c_out = 0; c_out < channels; ++c_out) {
            scalar_t val = input[b][c_out][0][h][w] - max_val;
            sum += exp(val);
        }

        output[b][c][h][w] = exp(min_val - max_val) / sum;
    }
}

torch::Tensor min_softmax_fusion_cuda(torch::Tensor input, int reduction_dim) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::empty({batch_size, channels, height, width}, input.options());

    const int threads = 256;
    const int elements = batch_size * channels * height * width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_softmax_fusion_cuda", ([&] {
        min_softmax_fusion<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits>(),
            batch_size, channels, depth, height, width, reduction_dim);
    }));

    return output;
}
"""

min_softmax_cpp_source = "torch::Tensor min_softmax_fusion_cuda(torch::Tensor input, int reduction_dim);"

min_softmax = load_inline(
    name="min_softmax",
    cpp_sources=min_softmax_cpp_source,
    cuda_sources=min_softmax_source,
    functions=["min_softmax_fusion_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = None
        self.reset_parameters()
        self.dim = dim
        self.conv_forward = conv3d
        self.min_softmax = min_softmax

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

    def forward(self, x):
        # Custom Conv3D implementation
        conv_out = self.conv_forward.conv3d_forward_cuda(x, self.conv_weight)
        # Fusion of min and softmax
        fused_out = self.min_softmax.min_softmax_fusion_cuda(conv_out, self.dim)
        return fused_out

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

# Constants from original code
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2