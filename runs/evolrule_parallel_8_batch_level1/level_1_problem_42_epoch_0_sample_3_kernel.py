import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Define and load the custom CUDA kernel
        self.max_pool2d = load_inline(
            name="max_pool2d",
            cuda_sources="""
                #include <torch/extension.h>
                #include <cuda_runtime.h>

                template <typename scalar_t>
                __global__ void max_pool2d_kernel(
                    const scalar_t* __restrict__ input,
                    scalar_t* __restrict__ output,
                    const int batch_size,
                    const int channels,
                    const int in_height,
                    const int in_width,
                    const int out_height,
                    const int out_width,
                    const int kernel_size,
                    const int stride,
                    const int padding,
                    const int dilation
                ) {
                    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (output_idx >= batch_size * channels * out_height * out_width) {
                        return;
                    }

                    const int w = output_idx % out_width;
                    const int h = (output_idx / out_width) % out_height;
                    const int c = (output_idx / (out_width * out_height)) % channels;
                    const int n = output_idx / (out_width * out_height * channels);

                    scalar_t max_val = -FLT_MAX;
                    int max_h = 0;
                    int max_w = 0;

                    // Apply padding to the input coordinates
                    int pad_h_start = -padding;
                    int pad_w_start = -padding;

                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_in = h * stride + kh * dilation + pad_h_start;
                            int w_in = w * stride + kw * dilation + pad_w_start;

                            // Check if the current position is within valid input boundaries
                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                scalar_t val = input[n * channels * in_height * in_width +
                                                     c * in_height * in_width +
                                                     h_in * in_width + w_in];

                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }

                    output[output_idx] = max_val;
                }

                torch::Tensor max_pool2d_forward(
                    torch::Tensor input,
                    int kernel_size,
                    int stride,
                    int padding,
                    int dilation
                ) {
                    const int batch_size = input.size(0);
                    const int channels = input.size(1);
                    const int in_height = input.size(2);
                    const int in_width = input.size(3);

                    // Calculate output dimensions
                    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
                    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

                    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

                    const int threads = 256;
                    const int elements = batch_size * channels * out_height * out_width;
                    const int blocks = (elements + threads - 1) / threads;

                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_forward", ([&] {
                        max_pool2d_kernel<scalar_t><<<blocks, threads>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size,
                            channels,
                            in_height,
                            in_width,
                            out_height,
                            out_width,
                            kernel_size,
                            stride,
                            padding,
                            dilation
                        );
                    }));

                    return output;
                }
            """,
            functions=["max_pool2d_forward"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool2d.max_pool2d_forward(
            x.cuda(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]