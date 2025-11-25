import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the fused convolution CUDA kernel
fused_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_convolution(
    const scalar_t* input,
    const scalar_t* depthwise_weights,
    const scalar_t* pointwise_weights,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int b = blockIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.z * blockDim.x + threadIdx.x;

    if (h_out >= input_height || w_out >= input_width) return;

    extern __shared__ float shared_depthwise[];

    // Compute depthwise for all channels
    for (int c = 0; c < in_channels; ++c) {
        scalar_t sum = 0.0;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int h_in = h_out + i - padding;
                int w_in = w_out + j - padding;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = b * in_channels * input_height * input_width +
                                    c * input_height * input_width +
                                    h_in * input_width + w_in;
                    int dw_weight_idx = c * kernel_size * kernel_size + i * kernel_size + j;
                    sum += input[input_idx] * depthwise_weights[dw_weight_idx];
                }
            }
        }
        shared_depthwise[c] = sum;
    }
    __syncthreads();

    // Compute pointwise for all output channels
    for (int k = 0; k < out_channels; ++k) {
        scalar_t sum = 0.0;
        for (int c = 0; c < in_channels; ++c) {
            int pw_weight_idx = k * in_channels + c;
            sum += shared_depthwise[c] * pointwise_weights[pw_weight_idx];
        }
        // Write to output
        int output_idx = b * out_channels * input_height * input_width +
                         k * input_height * input_width +
                         h_out * input_width + w_out;
        output[output_idx] = sum;
    }
}

// A wrapper function to call the kernel
torch::Tensor fused_convolution_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weights,
    torch::Tensor pointwise_weights,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    auto output = torch::empty({batch_size, out_channels, input_height, input_width}, input.options());

    // Define block and grid dimensions
    int block_x = 16;
    int block_y = 16;
    dim3 threads(block_x, block_y, 1);
    dim3 blocks(
        batch_size,
        (input_height + block_y - 1) / block_y,
        (input_width + block_x - 1) / block_x
    );

    // Calculate shared memory size
    size_t shared_size = in_channels * sizeof(float);

    // Launch the kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_convolution", ([&] {
        fused_convolution<scalar_t><<<blocks, threads, shared_size>>>(
            input.data<scalar_t>(),
            depthwise_weights.data<scalar_t>(),
            pointwise_weights.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));

    return output;
}
"""

# Define the C++ headers and source
cpp_source = """
torch::Tensor fused_convolution_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weights,
    torch::Tensor pointwise_weights,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);
"""

cuda_source = fused_convolution_source

# Compile the CUDA extension
fused_convolution_op = load_inline(
    name="fused_convolution",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_convolution_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize depthwise and pointwise weights
        self.depthwise_weight = nn.Parameter(torch.empty(
            (in_channels, 1, kernel_size, kernel_size)
        ))
        self.pointwise_weight = nn.Parameter(torch.empty(
            (out_channels, in_channels, 1, 1)
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize parameters
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pointwise_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Reshape pointwise weights to 2D (out_channels x in_channels)
        pointwise_weights_reshaped = self.pointwise_weight.view(
            self.out_channels, self.in_channels
        ).contiguous()
        
        # Call the fused CUDA kernel
        output = fused_convolution_op.fused_convolution_cuda(
            x,
            self.depthwise_weight,
            pointwise_weights_reshaped,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output