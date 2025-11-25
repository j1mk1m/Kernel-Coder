import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel code for adding bias and summing over channels
fused_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_sum_with_bias_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size) {
    int batch_idx = blockIdx.x;
    const scalar_t* input_row = input + batch_idx * channels * spatial_size;
    const scalar_t* bias_ptr = bias;

    extern __shared__ scalar_t shared[];
    int tid = threadIdx.x;
    scalar_t sum = 0.0;

    // Each thread processes a channel in steps of blockDim.x
    for (int c = tid; c < channels; c += blockDim.x) {
        scalar_t val = input_row[c * spatial_size] + bias_ptr[c];
        sum += val;
    }

    // Write partial sum to shared memory
    shared[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx * spatial_size] = shared[0];
    }
}

std::tuple<torch::Tensor> fused_sum_with_bias(
    torch::Tensor input,
    torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int spatial_size = input.size(2) * input.size(3) * input.size(4);
    const int output_size = spatial_size;

    torch::Tensor output = torch::zeros({batch_size, 1, 1, 1, 1}, input.options());

    const int block_size = 256;
    const int shared_size = block_size * sizeof(float);

    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_sum_with_bias", ([&] {
        fused_sum_with_bias_kernel<scalar_t><<<batch_size, block_size, shared_size>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            spatial_size
        );
    }));

    return std::make_tuple(output);
}
"""

fused_sum_cpp_source = R"""std::tuple<torch::Tensor> fused_sum_with_bias(
    torch::Tensor input,
    torch::Tensor bias);
"""

# Compile the fused kernel
fused_sum = load_inline(
    name="fused_sum",
    cpp_sources=fused_sum_cpp_source,
    cuda_sources=fused_sum_source,
    functions=["fused_sum_with_bias"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_sum = fused_sum

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        output = self.fused_sum.fused_sum_with_bias(x, self.bias)
        return output[0]  # Extract the tensor from the tuple