import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for custom convolution
custom_conv_source = """
#include <torch/extension.h>

#define IN_CHANNELS 3
#define OUT_CHANNELS 96
#define KERNEL_SIZE 11
#define STRIDE 4
#define PADDING 2
#define IN_HEIGHT 224
#define IN_WIDTH 224
#define OUT_HEIGHT 55
#define OUT_WIDTH 55
#define BATCH_SIZE 256

__global__ void custom_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
    float* output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= BATCH_SIZE * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH)
        return;

    // Compute indices
    int n = idx / (OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH);
    int remaining = idx % (OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH);
    int c_out = remaining / (OUT_HEIGHT * OUT_WIDTH);
    remaining = remaining % (OUT_HEIGHT * OUT_WIDTH);
    int y = remaining / OUT_WIDTH;
    int x = remaining % OUT_WIDTH;

    float sum = 0.0f;

    for (int c_in = 0; c_in < IN_CHANNELS; ++c_in) {
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int h_in = y * STRIDE + kh - PADDING;
                int w_in = x * STRIDE + kw - PADDING;

                if (h_in >= 0 && h_in < IN_HEIGHT &&
                    w_in >= 0 && w_in < IN_WIDTH) {
                    int input_offset = n * IN_CHANNELS * IN_HEIGHT * IN_WIDTH +
                                      c_in * IN_HEIGHT * IN_WIDTH +
                                      h_in * IN_WIDTH + w_in;
                    int weight_offset = c_out * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
                                       c_in * KERNEL_SIZE * KERNEL_SIZE +
                                       kh * KERNEL_SIZE + kw;
                    sum += input[input_offset] * weights[weight_offset];
                }
            }
        }
    }

    sum += bias[c_out]; // Add bias term

    int output_offset = n * OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH +
                       c_out * OUT_HEIGHT * OUT_WIDTH +
                       y * OUT_WIDTH + x;
    output[output_offset] = sum;
}

torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weights, torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int out_channels = weights.size(0);
    const int out_height = OUT_HEIGHT;
    const int out_width = OUT_WIDTH;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    const int num_elements = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 512;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv2d_cuda", ([&] {
        custom_conv2d<<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            weights.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>()
        );
    }));

    return output;
}
"""

custom_conv_cpp = """
torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weights, torch::Tensor bias);
"""

# Load the CUDA extension
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=[custom_conv_cpp],
    cuda_sources=[custom_conv_source],
    functions=["custom_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Initialize custom convolution parameters
        self.weight = nn.Parameter(torch.randn(OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE))
        self.bias = nn.Parameter(torch.randn(OUT_CHANNELS))

    def forward(self, x):
        return custom_conv.custom_conv2d_cuda(x, self.weight, self.bias)