import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>

torch::Tensor avg_pool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* input, float* output,
    int batch_size, int in_channels, int input_length,
    int kernel_size, int padding, int output_length) {

    int bid = blockIdx.y;
    int batch = bid / in_channels;
    int channel = bid % in_channels;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int o = tid;

    if (o >= output_length) return;

    int start = o - padding;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        int pos_in_padded = start + i;
        if (pos_in_padded < 0 || pos_in_padded >= input_length) continue;
        int in_offset = batch * in_channels * input_length +
                        channel * input_length +
                        pos_in_padded;
        sum += input[in_offset];
    }

    float avg = sum / static_cast<float>(kernel_size);
    int out_offset = batch * in_channels * output_length +
                     channel * output_length +
                     o;
    output[out_offset] = avg;
}

torch::Tensor avg_pool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    if (stride != 1) {
        AT_CHECK(false, "Stride must be 1 for this kernel");
    }

    auto output = torch::empty({batch_size, in_channels, output_length}, input.options());

    int threads_per_block = 256;
    dim3 blocks(
        (output_length + threads_per_block - 1) / threads_per_block,
        batch_size * in_channels
    );
    dim3 threads(threads_per_block);

    avg_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        kernel_size,
        padding,
        output_length
    );

    return output;
}
"""

avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["avg_pool1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = avg_pool_cuda.avg_pool1d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool(x, self.kernel_size, self.stride, self.padding)