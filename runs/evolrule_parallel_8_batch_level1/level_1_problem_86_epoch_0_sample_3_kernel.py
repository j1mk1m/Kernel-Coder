import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_convolution(
    const scalar_t* input,
    const scalar_t* depthwise_weights,
    const scalar_t* pointwise_weights,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    extern __shared__ float shared[];
    float* depthwise_vals = shared;
    float* pw_weights = depthwise_vals + in_channels;

    int batch = blockIdx.x / (output_height * output_width);
    int h = (blockIdx.x % (output_height * output_width)) / output_width;
    int w = (blockIdx.x % (output_height * output_width)) % output_width;

    if (batch >= batch_size || h >= output_height || w >= output_width)
        return;

    // Compute depthwise contributions
    for (int i = threadIdx.x; i < in_channels; i += blockDim.x) {
        float val = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h * stride + kh * dilation - padding;
                int w_in = w * stride + kw * dilation - padding;
                if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width)
                    continue;
                int input_offset = batch * in_channels * input_height * input_width +
                                   i * input_height * input_width +
                                   h_in * input_width + w_in;
                int dw_offset = i * kernel_size * kernel_size + kh * kernel_size + kw;
                val += input[input_offset] * depthwise_weights[dw_offset];
            }
        }
        depthwise_vals[i] = val;
    }

    __syncthreads();

    // Load pointwise weights into shared memory
    for (int idx = threadIdx.x; idx < out_channels * in_channels; idx += blockDim.x) {
        int o = idx / in_channels;
        int i = idx % in_channels;
        pw_weights[idx] = pointwise_weights[o * in_channels + i];
    }

    __syncthreads();

    // Compute output for each o
    for (int o = threadIdx.x; o < out_channels; o += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < in_channels; i++) {
            sum += depthwise_vals[i] * pw_weights[o * in_channels + i];
        }
        int output_offset = batch * out_channels * output_height * output_width +
                           o * output_height * output_width +
                           h * output_width + w;
        output[output_offset] = sum;
    }
}

torch::Tensor fused_convolution_cuda(torch::Tensor x,
                                     torch::Tensor depthwise_weights,
                                     torch::Tensor pointwise_weights,
                                     int batch_size,
                                     int in_channels,
                                     int out_channels,
                                     int kernel_size,
                                     int stride,
                                     int padding,
                                     int dilation,
                                     int input_height,
                                     int input_width,
                                     int output_height,
                                     int output_width) {
    auto device = x.device();
    AT_ASSERT(device.type() == torch::kCUDA);

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, x.options());

    int threadsPerBlock = 256;
    int blocks = batch_size * output_height * output_width;
    int sm_size = (in_channels + out_channels * in_channels) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_convolution_cuda", ([&] {
        auto input_data = x.data_ptr<scalar_t>();
        auto dw_data = depthwise_weights.data_ptr<scalar_t>();
        auto pw_data = pointwise_weights.data_ptr<scalar_t>();
        auto output_data = output.data_ptr<scalar_t>();

        fused_convolution<scalar_t><<<blocks, threadsPerBlock, sm_size>>>(
            input_data,
            dw_data,
            pw_data,
            output_data,
            batch_size,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            input_height,
            input_width,
            output_height,
            output_width
        );
    }));

    return output;
}
"""

fused_conv = load_inline(
    name="fused_conv",
    cuda_sources=fused_conv_source,
    functions=["fused_convolution_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.fused_conv = fused_conv

    def forward(self, x):
        stride = self.depthwise.stride[0]
        padding = self.depthwise.padding[0]
        dilation = self.depthwise.dilation[0]
        kernel_size = self.depthwise.kernel_size[0]
        in_channels = self.depthwise.in_channels
        out_channels = self.pointwise.out_channels

        N, C_in, H, W = x.shape
        output_height = int((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        output_width = int((W + 2*padding - dilation*(kernel_size-1) -1)/stride + 1)

        return self.fused_conv.fused_convolution_cuda(
            x,
            self.depthwise.weight,
            self.pointwise.weight,
            N,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            H,
            W,
            output_height,
            output_width
        )