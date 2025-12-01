import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 1D
maxpool_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool_1d_kernel(const float* input, float* output, int batch_size, int channels, int length, int kernel_size, int stride, int padding, int dilation) {
    int b = blockIdx.x / (channels * (length / stride));
    int c = (blockIdx.x / (length / stride)) % channels;
    int l = blockIdx.x % (length / stride);
    int start = l * stride - padding + dilation * (kernel_size - 1);
    int end = start + kernel_size * dilation;

    float max_val = -FLT_MAX;
    for (int i = start; i < end && i < length; i += dilation) {
        max_val = fmaxf(max_val, input[b * channels * length + c * length + i]);
    }

    output[b * channels * (length / stride) + c * (length / stride) + l] = max_val;
}

torch::Tensor maxpool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto length = input.size(2);
    auto output_length = (length + padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output = torch::zeros({batch_size, channels, output_length}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * output_length + block_size - 1) / block_size;

    maxpool_1d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, length, kernel_size, stride, padding, dilation);

    return output;
}
"""

maxpool_1d_cpp_source = (
    "torch::Tensor maxpool_1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for Max Pooling 1D
maxpool_1d = load_inline(
    name="maxpool_1d",
    cpp_sources=maxpool_1d_cpp_source,
    cuda_sources=maxpool_1d_source,
    functions=["maxpool_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool_1d.maxpool_1d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)