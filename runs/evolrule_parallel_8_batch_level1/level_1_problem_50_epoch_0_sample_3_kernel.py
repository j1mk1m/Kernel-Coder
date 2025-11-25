import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for custom convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define IN_CHANNELS 3
#define OUT_CHANNELS 96
#define KERNEL_SIZE 11
#define STRIDE 4
#define PADDING 2
#define INPUT_HEIGHT 224
#define INPUT_WIDTH 224
#define OUTPUT_HEIGHT 55
#define OUTPUT_WIDTH 55

__global__ void custom_conv2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) {
        return;
    }

    int n = idx / (OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH);
    int rest = idx % (OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH);
    int c_out = rest / (OUTPUT_HEIGHT * OUTPUT_WIDTH);
    int h_out_w_out = rest % (OUTPUT_HEIGHT * OUTPUT_WIDTH);
    int h_out = h_out_w_out / OUTPUT_WIDTH;
    int w_out = h_out_w_out % OUTPUT_WIDTH;

    float sum = 0.0f;

    for (int c_in = 0; c_in < IN_CHANNELS; ++c_in) {
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                int h_in_padded = h_out * STRIDE + kh;
                int w_in_padded = w_out * STRIDE + kw;

                int h_in = h_in_padded - PADDING;
                int w_in = w_in_padded - PADDING;

                if (h_in < 0 || h_in >= INPUT_HEIGHT ||
                    w_in < 0 || w_in >= INPUT_WIDTH) {
                    continue;
                }

                int input_offset = n * IN_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH +
                                   c_in * INPUT_HEIGHT * INPUT_WIDTH +
                                   h_in * INPUT_WIDTH + w_in;

                int weight_offset = c_out * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE +
                                    c_in * KERNEL_SIZE * KERNEL_SIZE +
                                    kh * KERNEL_SIZE + kw;

                sum += input[input_offset] * weights[weight_offset];
            }
        }
    }

    int output_offset = n * OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH +
                        c_out * OUTPUT_HEIGHT * OUTPUT_WIDTH +
                        h_out * OUTPUT_WIDTH + w_out;

    output[output_offset] = sum;
}

torch::Tensor custom_conv2d(
    torch::Tensor input,
    torch::Tensor weights) {

    int batch_size = input.size(0);

    auto output = torch::zeros({batch_size, OUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH}, input.options());

    int num_elements = batch_size * OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH;
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    custom_conv2d_forward<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size
    );

    return output;
}
"""

cpp_source = """
torch::Tensor custom_conv2d(
    torch::Tensor input,
    torch::Tensor weights);
"""

# Compile the CUDA extension
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=cpp_source,
    cuda_sources=convolution_source,
    functions=["custom_conv2d"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the convolution weights as a learnable parameter
        self.weight = nn.Parameter(torch.randn(96, 3, 11, 11, requires_grad=True))

    def forward(self, x):
        return custom_conv.custom_conv2d(x, self.weight)