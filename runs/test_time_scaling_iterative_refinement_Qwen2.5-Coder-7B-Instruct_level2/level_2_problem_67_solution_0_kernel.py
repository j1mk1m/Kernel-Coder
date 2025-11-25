import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height, int width, int out_channels, int kernel_size) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batch_size * out_channels) return;

    int oc = n / (height * width);
    int oh = n % (height * width) / width;
    int ow = n % (height * width) % width;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh + kh - kernel_size / 2;
                int iw = ow + kw - kernel_size / 2;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int ii = ic * height * width + ih * width + iw;
                    int wi = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    atomicAdd(&output[n], input[ii] * weight[wi]);
                }
            }
        }
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height, width, out_channels, kernel_size);

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


# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = 0.5 * input[idx] * (1.0 + tanh(sqrt(2.0 / M_PI) * (input[idx] + 0.044715)));
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), size);

    return output;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for GELU activation
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.gelu = gelu

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight)
        x = self.gelu.gelu_cuda(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]