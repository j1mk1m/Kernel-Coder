import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for logsumexp and ReLU
logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void logsumexp_relu_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= (batch_size * depth * height * width)) {
        return;
    }
    
    int batch = idx / (depth * height * width);
    int rem = idx % (depth * height * width);
    int d = rem / (height * width);
    int rem2 = rem % (height * width);
    int h = rem2 / width;
    int w = rem2 % width;
    
    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        int input_offset = batch * in_channels * depth * height * width +
                           c * depth * height * width +
                           d * height * width +
                           h * width +
                           w;
        float val = input[input_offset];
        sum += expf(val);
    }
    
    float logsum = logf(sum);
    float result = (logsum > 0) ? logsum : 0.0f;
    
    int output_offset = batch * depth * height * width +
                        d * height * width +
                        h * width +
                        w;
    output[output_offset] = result;
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    input = input.contiguous();
    auto output_size = input.sizes().vec();
    output_size[1] = 1;
    auto output = torch::empty(output_size, input.options());
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int num_threads = batch_size * depth * height * width;
    const int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;
    
    logsumexp_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        depth,
        height,
        width
    );
    
    return output;
}
"""

logsumexp_relu_cpp_source = """
#include <torch/extension.h>

torch::Tensor logsumexp_relu_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
logsumexp_relu = load_inline(
    name="logsumexp_relu",
    cpp_sources=logsumexp_relu_cpp_source,
    cuda_sources=logsumexp_relu_source,
    functions=["logsumexp_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.logsumexp_relu = logsumexp_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.logsumexp_relu.logsumexp_relu_cuda(x)
        return x