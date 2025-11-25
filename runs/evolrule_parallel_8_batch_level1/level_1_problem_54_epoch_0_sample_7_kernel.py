import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA error checking macro
CUDA_ERROR_CHECK = """
#define CUDA_KERNEL_CHECK()                               \\
    do {                                                 \\
        cudaError_t e = cudaGetLastError();               \\
        if (e != cudaSuccess) {                           \\
            printf("CUDA error: %s at %s:%d\\n",          \\
                   cudaGetErrorString(e), __FILE__, __LINE__); \\
            exit(-1);                                    \\
        }                                                \\
    } while (0)
"""

# Custom 3D convolution kernel implementation
conv3d_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

{CUDA_ERROR_CHECK}

template <typename scalar_t>
__global__ void conv3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int kernel_depth,
    const int kernel_height,
    const int kernel_width,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const scalar_t* __restrict__ bias
) {{
    // Calculate output dimensions
    const int out_depth = (in_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
    const int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    // Thread indices
    int output_idx = blockIdx.z * (gridDim.x * blockDim.x) + blockIdx.y * blockDim.y + threadIdx.x;
    int channel_out = blockIdx.x;
    int batch = blockIdx.z / (out_depth * out_height * out_width);

    // Compute output spatial indices
    int ow = output_idx % out_width;
    int oh = (output_idx / out_width) % out_height;
    int od = (output_idx / (out_width * out_height)) % out_depth;

    // Convert to input spatial indices with padding and stride
    int id_start = od * stride - padding;
    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;

    // Initialize output value
    scalar_t sum = 0.0;

    // Iterate over kernel dimensions
    for (int kd = 0; kd < kernel_depth; ++kd) {{
        int id_pad = id_start + dilation * kd;
        if (id_pad < 0 || id_pad >= in_depth) continue;

        for (int kh = 0; kh < kernel_height; ++kh) {{
            int ih_pad = ih_start + dilation * kh;
            if (ih_pad < 0 || ih_pad >= in_height) continue;

            for (int kw = 0; kw < kernel_width; ++kw) {{
                int iw_pad = iw_start + dilation * kw;
                if (iw_pad < 0 || iw_pad >= in_width) continue;

                // Iterate over input channels per group
                for (int c = 0; c < in_channels / groups; ++c) {{
                    int input_c = c + (channel_out / (out_channels / groups)) * (in_channels / groups);
                    int input_offset = batch * in_channels * in_depth * in_height * in_width +
                                      input_c * in_depth * in_height * in_width +
                                      id_pad * in_height * in_width +
                                      ih_pad * in_width +
                                      iw_pad;

                    int weight_offset = (channel_out % (out_channels / groups)) * (in_channels / groups) * kernel_depth * kernel_height * kernel_width +
                                       c * kernel_depth * kernel_height * kernel_width +
                                       kd * kernel_height * kernel_width +
                                       kh * kernel_width +
                                       kw;

                    sum += input[input_offset] * weight[weight_offset];
                }}
            }}
        }}
    }}

    if (bias != nullptr) {{
        sum += bias[channel_out];
    }}

    // Write output
    int output_offset = batch * out_channels * out_depth * out_height * out_width +
                       channel_out * out_depth * out_height * out_width +
                       od * out_height * out_width +
                       oh * out_width +
                       ow;

    output[output_offset] = sum;
}}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {{
    // Get input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Get kernel dimensions
    int out_channels = weight.size(0);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    // Compute output dimensions
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_depth - 1)) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_height - 1)) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_width - 1)) / stride + 1;

    // Output tensor
    auto output = torch::zeros({{batch_size, out_channels, out_depth, out_height, out_width}}, input.options());

    // Number of threads per block
    const int threads = 256;

    // Determine grid dimensions
    int output_elements_per_channel = out_depth * out_height * out_width;
    dim3 blocks_per_batch(
        out_channels,  // x: output channels
        (output_elements_per_channel + threads - 1) / threads,  // y: blocks per spatial dimension
        batch_size  // z: batches multiplied by spatial elements
    );

    // Launch kernel
    conv3d_kernel<<<blocks_per_batch, threads>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride,
        padding,
        dilation,
        groups,
        bias.data_ptr<scalar_t>()
    );

    CUDA_KERNEL_CHECK();

    return output;
}}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
"""

# Compile the CUDA kernel
conv3d = load_inline(
    name="conv3d",
    cpp_sources=cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.bias_term = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_term is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_term, -bound, bound)

    def forward(self, x):
        # Convert kernel_size to tuple if needed
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        # Handle bias tensor
        bias = self.bias_term if self.bias else torch.empty(0)

        return conv3d.conv3d_cuda(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )