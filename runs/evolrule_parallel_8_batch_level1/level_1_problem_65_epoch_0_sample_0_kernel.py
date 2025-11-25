import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv2d_kernel(
    const float* input, 
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int output_height,
    int output_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * output_height * output_width) return;

    // Compute the indices
    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    float sum = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out - kh;
                int w_in = w_out - kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    // Compute weight index
                    int weight_offset = c_in * out_channels * kernel_h * kernel_w
                                      + c_out * kernel_h * kernel_w
                                      + kh * kernel_w + kw;
                    float w_val = weight[weight_offset];

                    // Compute input index
                    int input_offset = n * in_channels * input_height * input_width
                                      + c_in * input_height * input_width
                                      + h_in * input_width + w_in;
                    float in_val = input[input_offset];

                    sum += in_val * w_val;
                }
            }
        }
    }

    // Compute output index
    int output_offset = n * out_channels * output_height * output_width
                      + c_out * output_height * output_width
                      + h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w) {

    int batch_size = input.size(0);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = input_height + kernel_h - 1;
    int output_width = input_width + kernel_w - 1;

    auto output = torch::zeros(
        {batch_size, out_channels, output_height, output_width},
        torch::device("cuda").dtype(torch::kFloat32)
    );

    int num_threads = batch_size * out_channels * output_height * output_width;
    int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;

    transposed_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        output_height,
        output_width
    );

    return output;
}
"""

transposed_conv2d_cpp_source = (
    "torch::Tensor transposed_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int in_channels, int out_channels, int kernel_h, int kernel_w);"
)

# Compile the inline CUDA code for the transposed convolution
transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_source,
    functions=["transposed_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.randn(
                in_channels,
                out_channels,
                kernel_size[0],
                kernel_size[1],
                dtype=torch.float32,
            )
        )
        self.transposed_conv2d = transposed_conv2d

    def forward(self, x):
        # Move input to GPU
        x_gpu = x.cuda()
        # Compute the output using the CUDA kernel
        output_gpu = self.transposed_conv2d.transposed_conv2d_cuda(
            x_gpu,
            self.weight,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        # Move the output back to CPU
        return output_gpu.cpu()