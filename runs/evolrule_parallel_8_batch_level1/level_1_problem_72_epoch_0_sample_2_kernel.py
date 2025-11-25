import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import torch.nn.functional as F

# Custom CUDA kernel for ConvTranspose3d
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Helper function to compute output dimensions
std::vector<int64_t> compute_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups) {
    auto in_channels = input.size(1);
    auto batch_size = input.size(0);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);

    auto kernel_depth = kernel_size[0];
    auto kernel_height = kernel_size[1];
    auto kernel_width = kernel_size[2];

    auto stride_d = stride[0];
    auto stride_h = stride[1];
    auto stride_w = stride[2];

    auto padding_d = padding[0];
    auto padding_h = padding[1];
    auto padding_w = padding[2];

    auto output_padding_d = output_padding[0];
    auto output_padding_h = output_padding[1];
    auto output_padding_w = output_padding[2];

    auto out_depth = (input_depth - 1) * stride_d - 2 * padding_d +
                     kernel_depth + output_padding_d;
    auto out_height = (input_height - 1) * stride_h - 2 * padding_h +
                      kernel_height + output_padding_h;
    auto out_width = (input_width - 1) * stride_w - 2 * padding_w +
                     kernel_width + output_padding_w;

    return {batch_size, weight.size(1), out_depth, out_height, out_width};
}

// CUDA kernel for transposed 3D convolution
__global__ void conv_transpose_3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int groups,
    bool has_bias,
    const float* bias) {

    // Calculate output dimensions
    int out_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    int out_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    int out_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;

    // Thread indices
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_idx >= batch_size * out_channels * out_depth * out_height * out_width) {
        return;
    }

    // Compute output coordinates
    int w = output_idx % out_width;
    int h = (output_idx / out_width) % out_height;
    int d = (output_idx / (out_width * out_height)) % out_depth;
    int c_out = (output_idx / (out_width * out_height * out_depth)) % out_channels;
    int n = output_idx / (out_channels * out_depth * out_height * out_width);

    float val = 0.0;

    // Iterate over input channels and kernel
    for (int c_in_group = 0; c_in_group < in_channels / groups; ++c_in_group) {
        int c_in = c_in_group + (c_out / (out_channels / groups)) * (in_channels / groups);
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    // Compute input coordinates
                    int input_d = (d - kd + padding_d) / stride_d;
                    int input_h = (h - kh + padding_h) / stride_h;
                    int input_w = (w - kw + padding_w) / stride_w;

                    // Check validity
                    if (input_d < 0 || input_d >= input_depth) continue;
                    if (input_h < 0 || input_h >= input_height) continue;
                    if (input_w < 0 || input_w >= input_width) continue;

                    // Accumulate the contribution
                    int weight_offset = (c_out * kernel_depth * kernel_height * kernel_width +
                                        kd * kernel_height * kernel_width +
                                        kh * kernel_width + kw) * (in_channels / groups) + c_in_group;
                    val += input[n * in_channels * input_depth * input_height * input_width +
                                c_in * input_depth * input_height * input_width +
                                input_d * input_height * input_width +
                                input_h * input_width + input_w] *
                            weight[weight_offset];
                }
            }
        }
    }

    if (has_bias) {
        val += bias[c_out];
    }

    output[output_idx] = val;
}

// Python entry point
torch::Tensor conv_transpose_3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool has_bias) {

    // Check dimensions and types
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D tensor");
    TORCH_CHECK(input.type().is_cuda(), "Input must be on GPU");
    TORCH_CHECK(weight.type().is_cuda(), "Weight must be on GPU");

    // Compute output size
    auto output_size = compute_output_size(input, weight, kernel_size, stride, padding, output_padding, groups);

    auto output = torch::empty(output_size, input.options());

    // Launch kernel
    int threads_per_block = 256;
    int num_elements = output.numel();
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose_3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        weight.size(1),
        input.size(2),
        input.size(3),
        input.size(4),
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        output_padding[0],
        output_padding[1],
        output_padding[2],
        groups,
        has_bias,
        bias.data_ptr<float>());

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_3d_cpp_source = """
torch::Tensor conv_transpose_3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool has_bias);
"""

# Compile the custom CUDA operator
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=[conv_transpose_3d_cpp_source],
    cuda_sources=[conv_transpose_3d_source],
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_70,code=sm_70"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), 
                 padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.has_bias = bias

        # Initialize weights and bias similar to PyTorch's ConvTranspose3d
        kernel_depth, kernel_height, kernel_width = kernel_size
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, 
            kernel_depth, kernel_height, kernel_width))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters (using default PyTorch initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Bind the custom CUDA function
        self.conv_transpose_3d = conv_transpose_3d

    def forward(self, x):
        # Ensure inputs are on the correct device
        x = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda() if self.has_bias else torch.empty(0).cuda()

        return self.conv_transpose_3d.conv_transpose_3d_cuda(
            x, weight, bias,
            self.kernel_size, self.stride, self.padding,
            self.output_padding, self.groups, self.has_bias)