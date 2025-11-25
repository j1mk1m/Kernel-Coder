import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

avg_pool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(const float* input, float* output, int batch_size, int in_channels, int input_length, int kernel_size, int stride, int padding, int output_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * in_channels * output_length)
        return;

    int o = tid % output_length;
    int c = (tid / output_length) % in_channels;
    int b = tid / (output_length * in_channels);

    float sum = 0.0f;
    int start = o * stride - padding;
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = start + k;
        if (in_pos >= 0 && in_pos < input_length) {
            sum += input[ b * in_channels * input_length + c * input_length + in_pos ];
        }
    }
    output[ b * in_channels * output_length + c * output_length + o ] = sum / (float)kernel_size;
}

torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    input = input.contiguous();
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int output_length = ((input_length + 2 * padding - kernel_size) / stride) + 1;
    
    auto output = torch::zeros({batch_size, in_channels, output_length}, input.options());
    
    const int block_size = 256;
    int num_elements = batch_size * in_channels * output_length;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    avg_pool1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_length
    );

    return output;
}
"""

avg_pool1d_cpp_source = """
torch::Tensor avg_pool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cuda_sources=avg_pool1d_source,
    cpp_sources=avg_pool1d_cpp_source,
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
        self.avg_pool_cuda = avg_pool_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool_cuda.avg_pool1d_cuda(
            x, self.kernel_size, self.stride, self.padding
        )

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]