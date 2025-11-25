import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Define custom fused Convolution + ReLU + BiasAdd CUDA kernel
        fused_conv_relu_bias_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void fused_conv_relu_bias_forward_kernel(
            const torch::PackedTensorAccessor<scalar_t,4> input,
            const torch::PackedTensorAccessor<scalar_t,4> weight,
            const torch::PackedTensorAccessor<scalar_t,4> bias,
            torch::PackedTensorAccessor<scalar_t,4> output,
            int out_channels, int out_h, int out_w,
            int kernel_size, int padding, int stride) {

            const int B = blockIdx.z;
            const int C = blockIdx.y;
            const int H = blockIdx.x * blockDim.y + threadIdx.y;
            const int W = threadIdx.x;

            if (H >= out_h || W >= out_w) return;

            scalar_t sum = 0;
            for (int k = 0; k < input.size(1); ++k) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h_in = H * stride - padding + kh;
                        int w_in = W * stride - padding + kw;
                        if (h_in >= 0 && h_in < input.size(2) && w_in >= 0 && w_in < input.size(3)) {
                            sum += weight[C][k][kh][kw] * input[B][k][h_in][w_in];
                        }
                    }
                }
            }
            sum += bias[C][0][0];
            output[B][C][H][W] = sum > 0 ? sum : 0;
        }

        torch::Tensor fused_conv_relu_bias_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int kernel_size,
            int padding,
            int stride) {

            const int batch_size = input.size(0);
            const int in_channels = input.size(1);
            const int in_h = input.size(2);
            const int in_w = input.size(3);
            const int out_channels = weight.size(0);
            const int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
            const int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

            auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

            dim3 threads(32, 8);
            dim3 blocks(
                (out_h + threads.y - 1) / threads.y,
                (out_w),
                batch_size * out_channels
            );

            AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_relu_bias_forward", ([&] {
                fused_conv_relu_bias_forward_kernel<scalar_t><<<blocks, threads>>>(
                    input.packed_accessor<scalar_t,4>(),
                    weight.packed_accessor<scalar_t,4>(),
                    bias.packed_accessor<scalar_t,4>(),
                    output.packed_accessor<scalar_t,4>(),
                    out_channels, out_h, out_w,
                    kernel_size, padding, stride);
            }));

            cudaDeviceSynchronize();
            return output;
        }
        """

        fused_conv_relu_bias_cpp = (
            "torch::Tensor fused_conv_relu_bias_forward("
            "torch::Tensor input, "
            "torch::Tensor weight, "
            "torch::Tensor bias, "
            "int kernel_size, "
            "int padding, "
            "int stride);"
        )

        self.fused_conv_relu_bias = load_inline(
            name="fused_conv_relu_bias",
            cpp_sources=fused_conv_relu_bias_cpp,
            cuda_sources=fused_conv_relu_bias_source,
            functions=["fused_conv_relu_bias_forward"],
            verbose=True
        )

    def forward(self, x):
        # Get parameters from existing conv layer
        weight = self.conv.weight
        bias = self.bias
        padding = self.conv.padding[0]
        stride = self.conv.stride[0]
        kernel_size = self.conv.kernel_size[0]

        # Call fused kernel
        x = self.fused_conv_relu_bias.fused_conv_relu_bias_forward(
            x, weight, bias, kernel_size, padding, stride
        )
        return x

# Keep these functions unchanged
def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]