import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused logsumexp and ReLU CUDA kernel
fused_logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_logsumexp_relu_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * depth * height * width) return;

    // Compute spatial indices
    int w = idx % width;
    int rem = idx / width;
    int h = rem % height;
    rem = rem / height;
    int d = rem % depth;
    int b = rem / depth;

    scalar_t sum_exp = 0.0;

    for (int c = 0; c < channels; ++c) {
        int input_offset = b * channels * depth * height * width +
                          c * depth * height * width +
                          d * height * width +
                          h * width +
                          w;
        scalar_t val = input[input_offset];
        sum_exp += expf(static_cast<float>(val));
    }

    scalar_t log_sum = logf(sum_exp);
    scalar_t result = fmaxf(log_sum, 0.0f);

    // Output offset for (b, 0, d, h, w)
    int output_offset = b * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
    output[output_offset] = static_cast<scalar_t>(result);
}

// PyTorch wrapper function for fused kernel
torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input) {
    auto input_contig = input.contiguous();
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto output = torch::zeros({batch_size, 1, depth, height, width}, input.options());

    const int block_size = 256;
    int total_elements = batch_size * depth * height * width;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // Launch kernel with float specialization
    fused_logsumexp_relu_kernel<float><<<num_blocks, block_size>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );

    return output;
}
"""

# C++ header for the kernel
fused_logsumexp_relu_cpp_source = """
#include <torch/extension.h>

extern "C" {
    torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input);
}
"""

# Compile the fused CUDA kernel
fused_logsumexp_relu = load_inline(
    name="fused_logsumexp_relu",
    cpp_sources=fused_logsumexp_relu_cpp_source,
    cuda_sources=fused_logsumexp_relu_source,
    functions=["fused_logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fused_op = fused_logsumexp_relu  # CUDA operator handle

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.fused_op.fused_logsumexp_relu_cuda(x)
        return x

def get_inputs():
    batch_size = 4
    in_channels = 32
    depth, height, width = 32, 128, 128
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]

def get_init_inputs():
    return [32, 64, 3, 1, 1]  # in_channels, out_channels, kernel_size, stride, padding