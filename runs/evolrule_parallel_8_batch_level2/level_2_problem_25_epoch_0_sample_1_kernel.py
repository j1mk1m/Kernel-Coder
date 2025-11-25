import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for processing after convolution
process_after_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void process_after_conv(
    const float* input, float* output,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * W) return;

    int n = idx / (H * W);
    int rem = idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    float min_val = FLT_MAX;
    for (int c = 0; c < C; ++c) {
        int in_offset = n * C * H * W + c * H * W + h * W + w;
        float val = input[in_offset];
        if (val < min_val) {
            min_val = val;
        }
    }

    float val = tanhf(min_val);
    val = tanhf(val);

    int out_offset = n * H * W + h * W + w;
    output[out_offset] = val;
}

torch::Tensor process_after_conv_cuda(torch::Tensor input) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty({N, 1, H, W}, 
                              torch::dtype(input.dtype()).device(input.device()));

    const int total_elements = N * H * W;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    process_after_conv<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}
"""

process_after_conv_cpp_source = """
torch::Tensor process_after_conv_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for the custom processing
process_after_conv = load_inline(
    name="process_after_conv",
    cpp_sources=process_after_conv_cpp_source,
    cuda_sources=process_after_conv_source,
    functions=["process_after_conv_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.process_after_conv = process_after_conv  # Custom CUDA operator

    def forward(self, x):
        x = self.conv(x)
        x = self.process_after_conv.process_after_conv_cuda(x)
        return x