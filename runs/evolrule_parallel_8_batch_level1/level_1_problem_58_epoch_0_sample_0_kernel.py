import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_term = bias

        # Initialize weights and bias similar to ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(
            in_channels, 
            out_channels // groups, 
            *kernel_size
        ))
        self.reset_parameters()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Define the CUDA kernel
        conv_transpose3d_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <ATen/cuda/CUDAContext.h>

        template <typename scalar_t>
        __global__ void conv_transpose3d_kernel(
            const scalar_t* input,
            const scalar_t* weight,
            const scalar_t* bias,
            scalar_t* output,
            int batch_size,
            int in_channels,
            int out_channels,
            int depth_in,
            int height_in,
            int width_in,
            int depth_out,
            int height_out,
            int width_out,
            int kernel_d,
            int kernel_h,
            int kernel_w,
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
            bool has_bias
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_channels * depth_out * height_out * width_out) {
                return;
            }

            int b = idx / (out_channels * depth_out * height_out * width_out);
            int remainder = idx % (out_channels * depth_out * height_out * width_out);
            int c_out = remainder / (depth_out * height_out * width_out);
            remainder %= (depth_out * height_out * width_out);
            int d_out = remainder / (height_out * width_out);
            remainder %= (height_out * width_out);
            int h_out = remainder / width_out;
            int w_out = remainder % width_out;

            int group = c_out / (out_channels / groups);
            int c_out_in_group = c_out % (out_channels / groups);

            scalar_t sum = 0;

            for (int c_in = 0; c_in < in_channels / groups; ++c_in) {
                for (int kd = 0; kd < kernel_d; ++kd) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int d_in = (d_out - kd + padding_d - output_padding_d) / stride_d;
                            int h_in = (h_out - kh + padding_h - output_padding_h) / stride_h;
                            int w_in = (w_out - kw + padding_w - output_padding_w) / stride_w;

                            if (d_in >= 0 && d_in < depth_in &&
                                h_in >= 0 && h_in < height_in &&
                                w_in >= 0 && w_in < width_in) {
                                int in_c = c_in + group * (in_channels / groups);
                                int weight_offset = in_c * (out_channels / groups) * kernel_d * kernel_h * kernel_w +
                                                    c_out_in_group * kernel_d * kernel_h * kernel_w +
                                                    kd * kernel_h * kernel_w +
                                                    kh * kernel_w +
                                                    kw;

                                int input_offset = b * in_channels * depth_in * height_in * width_in +
                                                   in_c * depth_in * height_in * width_in +
                                                   d_in * height_in * width_in +
                                                   h_in * width_in +
                                                   w_in;

                                sum += input[input_offset] * weight[weight_offset];
                            }
                        }
                    }
                }
            }

            if (has_bias) {
                sum += bias[c_out];
            }

            int out_offset = b * out_channels * depth_out * height_out * width_out +
                            c_out * depth_out * height_out * width_out +
                            d_out * height_out * width_out +
                            h_out * width_out +
                            w_out;

            output[out_offset] = sum;
        }

        at::Tensor conv_transpose3d_cuda(
            at::Tensor input,
            at::Tensor weight,
            at::Tensor bias,
            at::IntArrayRef stride,
            at::IntArrayRef padding,
            at::IntArrayRef output_padding,
            int64_t groups
        ) {
            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto depth_in = input.size(2);
            const auto height_in = input.size(3);
            const auto width_in = input.size(4);

            const auto kernel_d = weight.size(2);
            const auto kernel_h = weight.size(3);
            const auto kernel_w = weight.size(4);

            const auto stride_d = stride[0];
            const auto stride_h = stride[1];
            const auto stride_w = stride[2];

            const auto padding_d = padding[0];
            const auto padding_h = padding[1];
            const auto padding_w = padding[2];

            const auto output_padding_d = output_padding[0];
            const auto output_padding_h = output_padding[1];
            const auto output_padding_w = output_padding[2];

            const auto out_channels_per_group = weight.size(1);
            const auto out_channels = out_channels_per_group * groups;

            const auto depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
            const auto height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
            const auto width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

            auto output = at::empty({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

            const int threads = 512;
            const int num_elements = batch_size * out_channels * depth_out * height_out * width_out;
            const int blocks = (num_elements + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_cuda", ([&] {
                conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(),
                    weight.data<scalar_t>(),
                    bias.defined() ? bias.data<scalar_t>() : nullptr,
                    output.data<scalar_t>(),
                    batch_size, in_channels, out_channels,
                    depth_in, height_in, width_in,
                    depth_out, height_out, width_out,
                    kernel_d, kernel_h, kernel_w,
                    stride_d, stride_h, stride_w,
                    padding_d, padding_h, padding_w,
                    output_padding_d, output_padding_h, output_padding_w,
                    groups,
                    bias.defined()
                );
            }));

            cudaDeviceSynchronize();
            return output;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("conv_transpose3d_cuda", &conv_transpose3d_cuda, "Custom ConvTranspose3d CUDA implementation");
        }
        """

        # Compile the CUDA code
        self.conv_transpose3d_cuda = load_inline(
            name="conv_transpose3d_cuda",
            cuda_sources=conv_transpose3d_source,
            functions=["conv_transpose3d_cuda"],
            verbose=True
        )

    def reset_parameters(self):
        # Weight initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d_cuda(
            x,
            self.weight,
            self.bias if self.bias_term else torch.empty(0, device=x.device),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )