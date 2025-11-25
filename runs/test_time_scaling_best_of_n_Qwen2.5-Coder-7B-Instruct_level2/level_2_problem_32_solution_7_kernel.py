import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.x; // Batch index
    int c_out = blockIdx.y; // Output channel index
    int h_out = blockIdx.z / width; // Output height index
    int w_out = blockIdx.z % width; // Output width index

    int h_in = h_out * stride + padding;
    int w_in = w_out * stride + padding;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_weight = kh;
                int w_weight = kw;
                int h_input = h_in - kh;
                int w_input = w_in - kw;

                if (h_input >= 0 && h_input < height && w_input >= 0 && w_input < width) {
                    sum += input[n * in_channels * height * width + c_in * height * width + h_input * width + w_input] *
                           weight[c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + h_weight * kernel_size + w_weight];
                }
            }
        }
    }

    output[n * out_channels * height * width + c_out * height * width + h_out * width + w_out] = sum;
}

torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, (height + 2 * padding - kernel_size) / stride + 1, (width + 2 * padding - kernel_size) / stride + 1}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * ((height + 2 * padding - kernel_size) / stride + 1) * ((width + 2 * padding - kernel_size) / stride + 1) + block_size - 1) / block_size;

    convolution_2d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Custom convolution kernel
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
        x = convolution.convolution_2d_cuda(x, weight, stride=1, padding=1)

        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum along channel dimension
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]