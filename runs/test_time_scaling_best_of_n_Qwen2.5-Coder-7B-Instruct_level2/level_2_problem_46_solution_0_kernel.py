import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int c_in = blockIdx.x;
    int h_out = blockIdx.w;
    int w_out = threadIdx.x;

    float sum = 0.0f;
    for (int k_h = 0; k_h < kernel_size; ++k_h) {
        for (int k_w = 0; k_w < kernel_size; ++k_w) {
            int h_in = h_out * 2 - k_h + kernel_size / 2;
            int w_in = w_out * 2 - k_w + kernel_size / 2;
            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                int i = n * in_channels * height * width + c_in * height * width + h_in * width + w_in;
                int j = c_in * kernel_size * kernel_size + k_h * kernel_size + k_w;
                sum += input[i] * weight[j];
            }
        }
    }

    if (w_out == 0) {
        atomicAdd(&output[n * out_channels * height * width + c_out * height * width + h_out * width], sum);
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 block_size(1, 1, 1);
    dim3 grid_size(out_channels, 1, batch_size);

    convolution_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x