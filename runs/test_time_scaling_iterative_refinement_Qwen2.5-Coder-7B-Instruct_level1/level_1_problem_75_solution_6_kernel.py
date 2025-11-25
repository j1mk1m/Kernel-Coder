import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D transposed convolution
transposed_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to perform 2D transposed convolution
__global__ void transposed_convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_height, int kernel_width, int stride_height, int stride_width, int padding_height, int padding_width, int dilation_height, int dilation_width, int groups) {
    int batch_id = blockIdx.x / (height_out * width_out);
    int out_channel_id = (blockIdx.x % (height_out * width_out)) / width_out;
    int out_height_id = (blockIdx.x % (height_out * width_out)) / width_out;
    int out_width_id = blockIdx.x % width_out;

    int in_channel_id = out_channel_id * groups;
    int in_height_id = out_height_id * stride_height - padding_height + dilation_height * (kernel_height - 1);
    int in_width_id = out_width_id * stride_width - padding_width + dilation_width * (kernel_width - 1);

    float sum = 0.0f;
    for (int k = 0; k < kernel_height; ++k) {
        for (int l = 0; l < kernel_width; ++l) {
            int in_h_idx = in_height_id + k * dilation_height;
            int in_w_idx = in_width_id + l * dilation_width;
            if (in_h_idx >= 0 && in_h_idx < height_in && in_w_idx >= 0 && in_w_idx < width_in) {
                for (int g = 0; g < groups; ++g) {
                    int in_channel_g_id = in_channel_id + g;
                    int in_channel_g_offset = in_channel_g_id * height_in * width_in;
                    int in_index = in_channel_g_offset + in_h_idx * width_in + in_w_idx;
                    int w_index = out_channel_id * groups * kernel_height * kernel_width + g * kernel_height * kernel_width + k * kernel_width + l;
                    sum += input[in_index] * weight[w_index];
                }
            }
        }
    }

    int out_index = batch_id * out_channels * height_out * width_out + out_channel_id * height_out * width_out + out_height_id * width_out + out_width_id;
    output[out_index] = sum;
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride_height, int stride_width, int padding_height, int padding_width, int dilation_height, int dilation_width, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto kernel_height = weight.size(2);
    auto kernel_width = weight.size(3);
    auto height_out = (height_in - 1) * stride_height - 2 * padding_height + dilation_height * (kernel_height - 1) + 1;
    auto width_out = (width_in - 1) * stride_width - 2 * padding_width + dilation_width * (kernel_width - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height_out * width_out + block_size - 1) / block_size;

    transposed_convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width, groups);

    return output;
}
"""

transposed_convolution_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride_height, int stride_width, int padding_height, int padding_width, int dilation_height, int dilation_width, int groups);"
)

# Compile the inline CUDA code for 2D transposed convolution
transposed_convolution = load_inline(
    name="transposed_convolution",
    cpp_sources=transposed_convolution_cpp_source,
    cuda_sources=transposed_convolution_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        stride_height, stride_width = stride
        padding_height, padding_width = padding
        dilation_height, dilation_width = dilation
        groups = groups
        weight = self.weight  # Assuming weight is defined elsewhere in the class
        return self.transposed_convolution.transposed_convolution_cuda(x, weight, stride_height, stride_width, padding_height, padding_width, dilation_height, dilation_width, groups)


# Example usage
if __name__ == "__main__":
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = (3, 5)
    height = 128
    width = 256
    stride = (2, 3)
    padding = (1, 2)
    dilation = (2, 1)
    groups = 4

    model = ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    y = model(x)
    print(y.shape)