import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights and bias similar to ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Define and load custom CUDA kernel
        self.conv_transpose2d = load_inline(
            name="conv_transpose2d",
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>

                template <typename scalar_t>
                __global__ void conv_transpose2d_kernel(
                    const scalar_t* input,
                    const scalar_t* weight,
                    scalar_t* output,
                    const scalar_t* __restrict__ bias,
                    int batch_size,
                    int in_channels,
                    int out_channels,
                    int input_height,
                    int input_width,
                    int kernel_h,
                    int kernel_w,
                    int stride_h,
                    int stride_w,
                    int pad_h,
                    int pad_w,
                    int output_height,
                    int output_width
                ) {{
                    // Calculate output coordinates
                    int n = blockIdx.x;
                    int c_out = blockIdx.y;
                    int h_out = threadIdx.y + blockDim.y * blockIdx.z;
                    int w_out = threadIdx.x + blockDim.x * blockIdx.x;

                    if (h_out >= output_height || w_out >= output_width) {{
                        return;
                    }}

                    scalar_t val = 0;
                    for (int c_in = 0; c_in < in_channels; ++c_in) {{
                        for (int kh = 0; kh < kernel_h; ++kh) {{
                            for (int kw = 0; kw < kernel_w; ++kw) {{
                                // Compute input coordinates
                                int h_in = h_out + pad_h - kh;
                                int w_in = w_out + pad_w - kw;
                                if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {{
                                    continue;
                                }}

                                // Index calculation for weight
                                int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                                c_in * kernel_h * kernel_w +
                                                kh * kernel_w + kw;

                                // Index calculation for input
                                int input_idx = n * in_channels * input_height * input_width +
                                                c_in * input_height * input_width +
                                                h_in * input_width + w_in;

                                val += weight[weight_idx] * input[input_idx];
                            }}
                        }}
                    }}

                    if (bias != nullptr) {{
                        val += bias[c_out];
                    }}

                    // Output index
                    int output_idx = n * out_channels * output_height * output_width +
                                    c_out * output_height * output_width +
                                    h_out * output_width + w_out;

                    output[output_idx] = val;
                }}

                at::Tensor conv_transpose2d_cuda(
                    at::Tensor input,
                    at::Tensor weight,
                    at::Tensor bias,
                    int stride_h,
                    int stride_w,
                    int pad_h,
                    int pad_w
                ) {{
                    const int batch_size = input.size(0);
                    const int in_channels = input.size(1);
                    const int input_height = input.size(2);
                    const int input_width = input.size(3);
                    const int out_channels = weight.size(0);
                    const int kernel_h = weight.size(2);
                    const int kernel_w = weight.size(3);

                    // Compute output dimensions
                    int output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_h;
                    int output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_w;

                    auto output = at::empty({{batch_size, out_channels, output_height, output_width}},
                                           input.options());

                    const dim3 threads(32, 32);  // Threads per block (x, y)
                    dim3 blocks(
                        (output_width + threads.x - 1) / threads.x,
                        (output_height + threads.y - 1) / threads.y,
                        batch_size * out_channels
                    );

                    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {{
                        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
                            input.data<scalar_t>(),
                            weight.data<scalar_t>(),
                            output.data<scalar_t>(),
                            bias.defined() ? bias.data<scalar_t>() : nullptr,
                            batch_size,
                            in_channels,
                            out_channels,
                            input_height,
                            input_width,
                            kernel_h,
                            kernel_w,
                            stride_h,
                            stride_w,
                            pad_h,
                            pad_w,
                            output_height,
                            output_width
                        );
                    }}));

                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {{
                        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
                    }}

                    return output;
                }}
            """,
            functions=["conv_transpose2d_cuda"],
            verbose=True
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda() if self.bias is not None else torch.empty(0).cuda()

        return self.conv_transpose2d.conv_transpose2d_cuda(
            input_tensor,
            weight,
            bias,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1]
        )

# Ensure get_inputs and get_init_inputs remain unchanged as per the problem statement
def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]