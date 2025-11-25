import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_conv_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N,
    int C_in,
    int C_out,
    int H,
    int W) {

    // Compute the block index (n, h, w)
    int block_idx = blockIdx.x;
    int n = block_idx / (H * W);
    int rem = block_idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    // Each thread corresponds to an output channel k
    int k = threadIdx.x;
    if (k >= C_out) return;

    // Load input channels into shared memory
    extern __shared__ scalar_t shared_input[];
    
    // Load input channels for this (n, h, w)
    if (threadIdx.x < C_in) {
        int c = threadIdx.x;
        int input_offset = n * C_in * H * W + 
                          c * H * W + 
                          h * W + 
                          w;
        shared_input[c] = input[input_offset];
    }
    __syncthreads();

    // Compute the dot product
    scalar_t sum = 0;
    for (int c = 0; c < C_in; ++c) {
        sum += shared_input[c] * weight[k * C_in + c]; // Weight is stored as row-major (C_out x C_in)
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[k];
    }

    // Compute output offset
    int output_offset = n * C_out * H * W +
                        k * H * W +
                        h * W +
                        w;

    output[output_offset] = sum;
}

// Host function to launch the kernel
torch::Tensor pointwise_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Get dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int C_out = weight.size(0);
    int H = input.size(2);
    int W = input.size(3);

    // Output tensor
    auto output = torch::empty({N, C_out, H, W}, input.options());

    // Number of blocks: N * H * W
    int num_blocks = N * H * W;
    int threads_per_block = C_out;

    // Allocate shared memory: C_in * sizeof(scalar_t)
    int shared_size = C_in * sizeof(float);

    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pointwise_conv_forward_cuda", ([&] {
        pointwise_conv_forward_kernel<scalar_t><<<num_blocks, threads_per_block, shared_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.defined() ? bias.data_ptr<scalar_t>() : nullptr),
            output.data_ptr<scalar_t>(),
            N, C_in, C_out, H, W);
    }));

    return output;
}
"""

pointwise_conv_cpp_source = """
#include <torch/extension.h>

torch::Tensor pointwise_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

pointwise_conv_cuda = load_inline(
    name="pointwise_conv_cuda",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.bias is not None:
            return pointwise_conv_cuda.pointwise_conv_forward_cuda(x, self.weight, self.bias)
        else:
            return pointwise_conv_cuda.pointwise_conv_forward_cuda(x, self.weight, torch.empty(0))