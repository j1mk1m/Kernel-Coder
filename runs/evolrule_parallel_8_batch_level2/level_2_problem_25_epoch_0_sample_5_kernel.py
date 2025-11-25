import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for min + two tanh operations
fused_min_tanh_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void fused_min_tanh_tanh_kernel(
    const float* input, float* output,
    int B, int C, int H, int W) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * H * W) return;

    int batch = tid / (H * W);
    int rem = tid % (H * W);
    int h = rem / W;
    int w = rem % W;

    float min_val = FLT_MAX;
    for (int c = 0; c < C; c++) {
        int input_idx = (batch * C + c) * H * W + h * W + w;
        float val = input[input_idx];
        if (val < min_val) {
            min_val = val;
        }
    }

    float result = tanhf(tanhf(min_val));

    int output_idx = batch * H * W + h * W + w;
    output[output_idx] = result;
}

torch::Tensor fused_min_tanh_tanh_cuda(torch::Tensor input) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty({B, 1, H, W}, input.options());

    int total_threads = B * H * W;
    const int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    fused_min_tanh_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );

    return output;
}
"""

fused_min_tanh_tanh_cpp = (
    "torch::Tensor fused_min_tanh_tanh_cuda(torch::Tensor input);"
)

# Load the CUDA extension
fused_min_tanh_tanh = load_inline(
    name="fused_min_tanh_tanh",
    cpp_sources=[fused_min_tanh_tanh_cpp],
    cuda_sources=[fused_min_tanh_tanh_source],
    functions=["fused_min_tanh_tanh_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_min_tanh_tanh = fused_min_tanh_tanh

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_min_tanh_tanh.fused_min_tanh_tanh_cuda(x)
        return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]