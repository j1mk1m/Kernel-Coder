import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused Conv3D + Softmax kernel
conv_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS 256

template <typename scalar_t>
__global__ void fused_conv_softmax_forward(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> weight,
    const torch::PackedTensorAccessor<scalar_t,1> bias,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int in_channels, int out_channels, int kernel_size,
    int batch_size, int depth, int height, int width,
    int out_depth, int out_height, int out_width) {

    const int B = blockIdx.z;
    const int C = blockIdx.y;
    const int D = blockIdx.x * blockDim.x + threadIdx.x;
    if (D >= out_depth) return;

    // Initialize output accumulation
    scalar_t sum = 0.0;
    scalar_t max_val = -INFINITY;

    // Compute convolution for this output position
    for (int d = 0; d < kernel_size; ++d) {
        for (int h = 0; h < kernel_size; ++h) {
            for (int w = 0; w < kernel_size; ++w) {
                for (int c = 0; c < in_channels; ++c) {
                    int id = D * kernel_size + d;
                    int ih = blockIdx.y * kernel_size + h;
                    int iw = blockIdx.x * kernel_size + w;
                    scalar_t val = input[B][c][id][ih][iw] * weight[C][c][d][h][w];
                    sum += val;
                }
            }
        }
    }
    sum += bias[C];

    // Compute max for softmax stabilization
    if (sum > max_val) max_val = sum;

    // Perform softmax denominator accumulation
    __shared__ scalar_t shared_sum[THREADS];
    shared_sum[threadIdx.x] = exp(sum - max_val);
    __syncthreads();

    // Reduction step for exp sum
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    scalar_t denom = (threadIdx.x == 0) ? shared_sum[0] : 0.0;
    __syncthreads();

    // Write output
    output[B][C][D][blockIdx.y][blockIdx.x] = exp(sum - max_val) / denom;
}

torch::Tensor fused_conv_softmax_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int out_channels, int out_depth, int out_height, int out_width) {

    auto output = torch::empty({input.size(0), out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threads(THREADS);
    dim3 blocks(out_width, out_height, input.size(0));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_softmax_forward", ([&] {
        fused_conv_softmax_forward<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            bias.packed_accessor<scalar_t,1>(),
            output.packed_accessor<scalar_t,5>(),
            input.size(1), out_channels, kernel_size,
            input.size(0), input.size(2), input.size(3), input.size(4),
            out_depth, out_height, out_width);
    }));

    return output;
}
"""

# Custom fused MaxPool3d kernel
maxpool_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_maxpool_forward(
    torch::PackedTensorAccessor<scalar_t,5> input,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int pool_size, int batch, int channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width) {

    int B = blockIdx.z;
    int C = blockIdx.y;
    int D = blockIdx.x * blockDim.x + threadIdx.x;
    if (D >= out_depth) return;

    scalar_t max_val = -INFINITY;
    for (int d = 0; d < pool_size; ++d) {
        for (int h = 0; h < pool_size; ++h) {
            for (int w = 0; w < pool_size; ++w) {
                int id = D * pool_size + d;
                int ih = blockIdx.y * pool_size + h;
                int iw = blockIdx.x * pool_size + w;
                scalar_t val = input[B][C][id][ih][iw];
                if (val > max_val) max_val = val;
            }
        }
    }
    output[B][C][D][blockIdx.y][blockIdx.x] = max_val;
}

torch::Tensor fused_maxpool_cuda(torch::Tensor input, int pool_size) {
    auto output_size = input.sizes().vec();
    output_size[2] /= pool_size;
    output_size[3] /= pool_size;
    output_size[4] /= pool_size;
    auto output = torch::empty(output_size, input.options());

    dim3 threads(256);
    dim3 blocks(output_size[4], output_size[3], output_size[0]);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_maxpool_forward", ([&] {
        fused_maxpool_forward<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            pool_size, input.size(0), input.size(1),
            input.size(2), input.size(3), input.size(4),
            output.size(2), output.size(3), output.size(4));
    }));

    return output;
}
"""

# Compile custom CUDA operators
conv_softmax_cpp = "torch::Tensor fused_conv_softmax_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int);"
conv_softmax = load_inline(name="conv_softmax", cpp_sources=conv_softmax_cpp, cuda_sources=conv_softmax_source, functions=["fused_conv_softmax_cuda"], verbose=False)

maxpool_cpp = "torch::Tensor fused_maxpool_cuda(torch::Tensor, int);"
maxpool = load_inline(name="maxpool", cpp_sources=maxpool_cpp, cuda_sources=maxpool_fused_source, functions=["fused_maxpool_cuda"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.pool_size = pool_kernel_size

        # Initialize convolution weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Save CUDA functions
        self.conv_softmax = conv_softmax
        self.maxpool = maxpool

    def forward(self, x):
        # Compute output spatial dimensions for convolution
        batch, _, in_depth, in_height, in_width = x.size()
        padding = (self.kernel_size - 1) // 2
        out_depth = (in_depth + 2*padding - self.kernel_size) // 1 + 1
        out_height = (in_height + 2*padding - self.kernel_size) // 1 + 1
        out_width = (in_width + 2*padding - self.kernel_size) // 1 + 1

        # Fused convolution + softmax
        conv_out = self.conv_softmax.fused_conv_softmax_cuda(
            x, self.weight, self.bias,
            self.kernel_size, self.weight.size(0),
            out_depth, out_height, out_width
        )

        # Fused max pooling (two passes)
        pool1 = self.maxpool.fused_maxpool_cuda(conv_out, self.pool_size)
        pool2 = self.maxpool.fused_maxpool_cuda(pool1, self.pool_size)

        return pool2

# Ensure initialization matches original model
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]