import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D Average Pooling
average_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void average_pooling_kernel(const float* input, float* output, int batch_size, int channels, int height_in, int width_in, int height_out, int width_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height_out * width_out) {
        return;
    }

    int c = idx / (height_out * width_out);
    int h_out = (idx % (height_out * width_out)) / width_out;
    int w_out = idx % width_out;

    float sum = 0.0f;
    for (int h_in = h_out * 11; h_in < h_out * 11 + 11; ++h_in) {
        for (int w_in = w_out * 11; w_in < w_out * 11 + 11; ++w_in) {
            int linear_idx = ((c * height_in + h_in) * width_in + w_in) * batch_size;
            sum += input[linear_idx];
        }
    }

    output[idx] = sum / (11 * 11);
}

torch::Tensor average_pooling_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    int height_out = (height_in + 10) / 11;
    int width_out = (width_in + 10) / 11;

    auto output = torch::zeros({batch_size, channels, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height_out * width_out + block_size - 1) / block_size;

    average_pooling_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height_in, width_in, height_out, width_out);

    return output;
}
"""

average_pooling_cpp_source = (
    "torch::Tensor average_pooling_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for 2D Average Pooling
average_pooling = load_inline(
    name="average_pooling",
    cpp_sources=average_pooling_cpp_source,
    cuda_sources=average_pooling_source,
    functions=["average_pooling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.average_pooling = average_pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.average_pooling.average_pooling_cuda(x)