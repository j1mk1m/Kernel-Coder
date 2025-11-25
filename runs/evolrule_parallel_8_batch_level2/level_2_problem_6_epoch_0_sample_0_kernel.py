import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

        # Initialize convolution parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # Custom Conv3D kernel with optimized memory access
        conv3d_kernel = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void conv3d_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int in_channels,
            const int out_channels,
            const int input_depth,
            const int input_height,
            const int input_width,
            const int kernel_size,
            const int output_depth,
            const int output_height,
            const int output_width,
            const int stride,
            const int padding) {

            const int output_spatial = output_depth * output_height * output_width;
            const int total_elements = batch_size * out_channels * output_spatial;
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid >= total_elements) return;

            const int w = tid % output_width;
            const int h = (tid / output_width) % output_height;
            const int d = (tid / (output_width * output_height)) % output_depth;
            const int oc = (tid / (output_spatial)) % out_channels;
            const int n = tid / (out_channels * output_spatial);

            scalar_t sum = __ldg(bias + oc);

            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kd = 0; kd < kernel_size; ++kd) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int id = d * stride + kd - padding;
                            int ih = h * stride + kh - padding;
                            int iw = w * stride + kw - padding;

                            if (id >= 0 && id < input_depth &&
                                ih >= 0 && ih < input_height &&
                                iw >= 0 && iw < input_width) {
                                const int input_offset = 
                                    n * in_channels * input_depth * input_height * input_width +
                                    ic * input_depth * input_height * input_width +
                                    id * input_height * input_width +
                                    ih * input_width +
                                    iw;

                                const int weight_offset = 
                                    oc * in_channels * kernel_size * kernel_size * kernel_size +
                                    ic * kernel_size * kernel_size * kernel_size +
                                    kd * kernel_size * kernel_size +
                                    kh * kernel_size +
                                    kw;

                                sum += __ldg(input + input_offset) * __ldg(weight + weight_offset);
                            }
                        }
                    }
                }
            }

            const int output_offset = 
                n * out_channels * output_depth * output_height * output_width +
                oc * output_depth * output_height * output_width +
                d * output_height * output_width +
                h * output_width +
                w;

            output[output_offset] = sum;
        }

        torch::Tensor conv3d_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int kernel_size,
            int stride=1,
            int padding=0) {

            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto input_depth = input.size(2);
            const auto input_height = input.size(3);
            const auto input_width = input.size(4);

            const auto out_channels = weight.size(0);

            const auto output_depth = (input_depth + 2 * padding - kernel_size) / stride + 1;
            const auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
            const auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

            auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

            const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
            const int threads_per_block = 256;
            const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

            AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_forward", ([&] {
                conv3d_kernel<scalar_t><<<blocks, threads_per_block>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, in_channels, out_channels,
                    input_depth, input_height, input_width,
                    kernel_size, output_depth, output_height, output_width,
                    stride, padding);
            }));

            cudaDeviceSynchronize();
            return output;
        }
        """

        self.conv3d_op = load_inline(
            name="custom_conv3d",
            cuda_sources=conv3d_kernel,
            functions=["conv3d_forward"],
            verbose=True
        )

    def forward(self, x):
        # Apply custom convolution with padding=1 and stride=1
        x = self.conv3d_op.conv3d_forward(
            x,
            self.weight,
            self.bias,
            self.kernel_size,
            padding=1  # Matches PyTorch Conv3d default
        )
        # Apply Softmax along channel dimension
        x = torch.softmax(x, dim=1)
        # Apply MaxPool3d twice
        x = self.pool1(x)
        x = self.pool2(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]