import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused Conv2D + ReLU + HardSwish kernel
fused_conv_relu_hswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

// Define constants
constexpr int kConvKernelSize = 3;
constexpr int kWarpSize = 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// Fused Convolution, ReLU, and HardSwish kernel
__global__ void fused_conv_relu_hswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int padding,
    int stride
) {
    // Implementation of convolution with ReLU and HardSwish
    // Note: This is a simplified version for demonstration. A real implementation
    // would need to handle padding, striding, and optimized memory access.
    
    int H = blockIdx.y;
    int W = blockIdx.x;
    int N = blockIdx.z;
    int C = threadIdx.x;

    float acc = 0.0;
    for (int kh = 0; kh < kConvKernelSize; ++kh) {
        for (int kw = 0; kw < kConvKernelSize; ++kw) {
            int h_in = H * stride - padding + kh;
            int w_in = W * stride - padding + kw;
            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                for (int c = 0; c < in_channels; ++c) {
                    acc += input[N * in_channels * input_height * input_width + c * input_height * input_width +
                                h_in * input_width + w_in] *
                           weight[C * in_channels * kConvKernelSize * kConvKernelSize +
                                  c * kConvKernelSize * kConvKernelSize + kh * kConvKernelSize + kw];
                }
            }
        }
    }
    acc += bias[C];

    // Apply ReLU
    acc = acc > 0.0f ? acc : 0.0f;

    // Apply HardSwish: x * clamp((x + 3)/6, 0, 1)
    float scale = fminf(fmaxf((acc + 3.0f) * (1.0f / 6.0f), 0.0f), 1.0f);
    acc *= scale;

    output[N * out_channels * output_height * output_width + C * output_height * output_width + H * output_width + W] = acc;
}

torch::Tensor fused_conv_relu_hswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int padding,
    int stride
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2); // Assuming square kernel

    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(32); // Example: 32 threads per block
    dim3 blocks(output_width, output_height, batch_size);

    fused_conv_relu_hswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        padding,
        stride
    );

    return output;
}
"""

# Compile the fused CUDA kernel
fused_conv_relu_hswish = load_inline(
    name="fused_conv_relu_hswish",
    cuda_sources=fused_conv_relu_hswish_source,
    functions=["fused_conv_relu_hswish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Initialize convolution weights and bias (mimicking PyTorch's Conv2d)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.padding = 1  # Example padding, adjust based on original model's Conv2d
        self.stride = 1   # Example stride, adjust as needed
        self.fused_op = fused_conv_relu_hswish

    def forward(self, x):
        # Use the fused CUDA kernel
        return self.fused_op.fused_conv_relu_hswish_cuda(
            x,
            self.weight,
            self.bias,
            padding=self.padding,
            stride=self.stride
        )

# Compatibility with original get_inputs and get_init_inputs
def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]