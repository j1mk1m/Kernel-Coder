import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_conv_inorm_div_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

template <typename scalar_t>
__global__ void fused_conv_inorm_div_forward(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    const torch::PackedTensorAccessor<scalar_t,1> bias,
    const torch::PackedTensorAccessor<scalar_t,1> inorm_weight,
    const torch::PackedTensorAccessor<scalar_t,1> inorm_bias,
    scalar_t divide_by,
    torch::PackedTensorAccessor<scalar_t,4> output
) {
    int batch = blockIdx.x;
    int out_h = blockIdx.y;
    int out_w = blockIdx.z;
    int out_c = threadIdx.x;

    // Compute output dimensions with padding
    int kernel_size = weight.size(2);
    int pad = kernel_size / 2;
    int in_h = input.size(2);
    int in_w = input.size(3);

    // Convolution computation
    scalar_t sum = bias[out_c];
    for (int in_c = 0; in_c < input.size(1); ++in_c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h = out_h - pad + kh;
                int w = out_w - pad + kw;
                if (h >= 0 && h < in_h && w >= 0 && w < in_w) {
                    sum += input[batch][in_c][h][w] * weight[out_c][in_c][kh][kw];
                }
            }
        }
    }

    // Instance normalization (per sample)
    __shared__ scalar_t shared_data[256]; // Size depends on out_channels
    shared_data[out_c] = sum;
    __syncthreads();

    // Compute mean and variance for the current channel
    scalar_t mean = 0.0, var = 0.0;
    for (int c = 0; c < output.size(1); ++c) {
        if (c == out_c) {
            mean += shared_data[c];
            var += shared_data[c] * shared_data[c];
        }
    }
    __syncthreads();

    mean /= output.size(1);
    var = sqrt(var / output.size(1) - mean * mean);

    // Normalize and divide
    output[batch][out_c][out_h][out_w] = 
        (shared_data[out_c] - mean) * inorm_weight[out_c] / (var + 1e-5) + inorm_bias[out_c];
    output[batch][out_c][out_h][out_w] /= divide_by;
}

torch::Tensor fused_conv_inorm_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor inorm_weight,
    torch::Tensor inorm_bias,
    float divide_by
) {
    int batch_size = input.size(0);
    int out_channels = weight.size(0);
    int out_height = input.size(2);
    int out_width = input.size(3);

    auto output = torch::empty_like(input);

    dim3 blocks(batch_size, out_height, out_width);
    dim3 threads(out_channels);

    fused_conv_inorm_div_forward<float><<<blocks, threads>>>(
        input.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        bias.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        inorm_weight.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        inorm_bias.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        divide_by,
        output.packed_accessor32<float,4,torch::RestrictPtrTraits>()
    );

    return output;
}
"""

# Load the fused kernel
fused_conv_inorm_div = load_inline(
    name="fused_conv_inorm_div",
    cuda_sources=fused_conv_inorm_div_source,
    functions=["fused_conv_inorm_div_cuda"],
    verbose=True
)

class FusedConvINormDiv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.inorm_weight = nn.Parameter(torch.ones(out_channels))
        self.inorm_bias = nn.Parameter(torch.zeros(out_channels))
        self.divide_by = divide_by

        # Initialize convolution parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_conv_inorm_div(
            x.cuda(), self.weight.cuda(), self.bias.cuda(),
            self.inorm_weight.cuda(), self.inorm_bias.cuda(),
            self.divide_by
        )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.layer = FusedConvINormDiv(in_channels, out_channels, kernel_size, divide_by)

    def forward(self, x):
        return self.layer(x)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]