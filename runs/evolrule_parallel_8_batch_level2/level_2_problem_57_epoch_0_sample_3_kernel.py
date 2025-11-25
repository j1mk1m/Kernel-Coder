import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused convolution + ReLU kernel
conv_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Fused convolution + ReLU kernel
at::Tensor conv_relu_cuda(
    const at::Tensor &input,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias,
    const std::vector<int64_t> &stride,
    const std::vector<int64_t> &padding,
    const std::vector<int64_t> &dilation,
    const int64_t groups) {

    // Get parameters
    auto input_data = input.contiguous();
    auto weight_data = weight.contiguous();
    auto bias_data = bias.has_value() ? bias.value().contiguous() : at::Tensor();
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);

    int64_t out_channels = weight.size(0);
    int64_t kernel_height = weight.size(2);
    int64_t kernel_width = weight.size(3);

    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t padding_h = padding[0];
    int64_t padding_w = padding[1];
    int64_t dilation_h = dilation[0];
    int64_t dilation_w = dilation[1];

    // Compute output dimensions
    int64_t output_height = (input_height + 2 * padding_h -
                            dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int64_t output_width = (input_width + 2 * padding_w -
                           dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    // Output tensor
    auto output = at::empty({batch_size, out_channels, output_height, output_width},
                           input.options());

    // Configure CUDA kernel
    const int threads = 256;
    dim3 blocks(
        (output_width + threads - 1) / threads,
        (output_height + threads - 1) / threads,
        batch_size * out_channels);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_relu_cuda", ([&] {
        using scalar_t = scalar_type;
        conv_relu_kernel<scalar_t><<<blocks, threads>>>(
            input_data.packed_accessor<scalar_t,4>(),
            weight_data.packed_accessor<scalar_t,4>(),
            bias_data.packed_accessor<scalar_t,1>(),
            output.packed_accessor<scalar_t,4>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups);
    }));

    return output;
}

// CUDA kernel implementation
template <typename scalar_t>
__global__ void conv_relu_kernel(
    const at::PackedTensorAccessor<scalar_t,4> input,
    const at::PackedTensorAccessor<scalar_t,4> weight,
    const at::PackedTensorAccessor<scalar_t,1> bias,
    at::PackedTensorAccessor<scalar_t,4> output,
    const int batch_size, const int in_channels, const int out_channels,
    const int input_height, const int input_width,
    const int kernel_height, const int kernel_width,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int dilation_h, const int dilation_w,
    const int groups) {

    // Calculate output coordinates
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z / out_channels;
    const int channel_out = blockIdx.z % out_channels;

    if (output_x >= output[0][0].size(1) || output_y >= output[0][0].size(0)) {
        return;
    }

    // Compute input region
    int in_x = output_x * stride_w - padding_w;
    int in_y = output_y * stride_h - padding_h;

    scalar_t sum = 0;
    if (bias.size(0) > 0) {
        sum = bias[channel_out];
    }

    // Iterate over kernel and input channels
    for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
        for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
            int in_x_actual = in_x + kernel_x * dilation_w;
            int in_y_actual = in_y + kernel_y * dilation_h;

            if (in_x_actual < 0 || in_x_actual >= input_width ||
                in_y_actual < 0 || in_y_actual >= input_height) {
                continue;
            }

            for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
                // Get weight index considering groups
                int weight_channel = in_channel / groups;
                int weight_idx = channel_out * in_channels/groups +
                                 weight_channel * kernel_height * kernel_width +
                                 kernel_y * kernel_width + kernel_x;
                
                sum += input[batch_id][in_channel][in_y_actual][in_x_actual] *
                       weight[weight_idx][in_channel % groups][kernel_y][kernel_x];
            }
        }
    }

    // Apply ReLU
    output[batch_id][channel_out][output_y][output_x] = fmax(scalar_t(0), sum);
}
"""

# Define the fused HardSwish kernel (clamp + multiplication)
hardswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_hardswish_kernel(
    const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float clamped = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
        output[idx] = x * clamped;
    }
}

at::Tensor fused_hardswish_cuda(at::Tensor input) {
    auto output = at::empty_like(input);
    int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fused_hardswish_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                               output.data_ptr<float>(), size);
    return output;
}
"""

# Compile the fused convolution + ReLU kernel
conv_relu_cpp = load_inline(
    name="conv_relu",
    cpp_sources="at::Tensor conv_relu_cuda(const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const std::vector<int64_t>&, const int64_t);",
    cuda_sources=conv_relu_source,
    functions=["conv_relu_cuda"],
    verbose=True
)

# Compile the fused HardSwish kernel
hardswish_cpp = load_inline(
    name="fused_hardswish",
    cpp_sources="at::Tensor fused_hardswish_cuda(at::Tensor);",
    cuda_sources=hardswish_source,
    functions=["fused_hardswish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)
        self.groups = 1

        # Convolution weights (we'll manually manage parameters)
        weight_shape = (out_channels, in_channels // self.groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape, device='cuda'))
        self.bias = nn.Parameter(torch.randn(out_channels, device='cuda'))

        # Register custom operators
        self.conv_relu = conv_relu_cpp
        self.hardswish = hardswish_cpp

    def forward(self, x):
        # Custom fused convolution + ReLU
        x = self.conv_relu.conv_relu_cuda(
            x.cuda(),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        # Fused HardSwish
        x = self.hardswish.fused_hardswish_cuda(x)
        return x

# Ensure the input generation matches the original dimensions
def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]