import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int output_padding,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) {
        return;
    }

    int w = idx % output_width;
    int h = (idx / output_width) % output_height;
    int d = (idx / (output_width * output_height)) % output_depth;
    int oc = (idx / (output_width * output_height * output_depth)) % out_channels;
    int b = idx / (output_width * output_height * output_depth * out_channels);

    float sum = 0.0;

    int out_channels_per_group = out_channels / groups;
    int oc_in_group = oc % out_channels_per_group;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int d_in = (d - kd + padding) / stride - output_padding;
                    int h_in = (h - kh + padding) / stride - output_padding;
                    int w_in = (w - kw + padding) / stride - output_padding;

                    if (d_in < 0 || d_in >= input_depth ||
                        h_in < 0 || h_in >= input_height ||
                        w_in < 0 || w_in >= input_width) {
                        continue;
                    }

                    int weight_offset = ic * out_channels_per_group * kernel_d * kernel_h * kernel_w +
                                        oc_in_group * kernel_d * kernel_h * kernel_w +
                                        kd * kernel_h * kernel_w +
                                        kh * kernel_w +
                                        kw;

                    float w_val = weight[weight_offset];

                    int input_offset = b * in_channels * input_depth * input_height * input_width +
                                       ic * input_depth * input_height * input_width +
                                       d_in * input_height * input_width +
                                       h_in * input_width +
                                       w_in;
                    float in_val = input[input_offset];

                    sum += w_val * in_val;
                }
            }
        }
    }

    int output_offset = b * out_channels * output_depth * output_height * output_width +
                        oc * output_depth * output_height * output_width +
                        d * output_height * output_width +
                        h * output_width +
                        w;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int output_padding,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int groups
) {
    auto device = input.device();
    TORCH_CHECK(weight.device() == device, "Input and weight must be on the same device.");

    auto options = torch::TensorOptions().dtype(input.dtype()).device(device);
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, options);

    int num_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv_transpose3d_kernel<<<num_blocks, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        stride,
        padding,
        output_padding,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int output_padding,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int groups
);
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weight
        kernel_shape = (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(kernel_shape, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Compile the CUDA kernel
        self.conv_transpose3d_cuda = load_inline(
            name="conv_transpose3d_cuda",
            cuda_sources=cuda_source,
            cpp_sources=cpp_source,
            functions=["conv_transpose3d_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_depth = x.size(2)
        input_height = x.size(3)
        input_width = x.size(4)
        batch_size = x.size(0)

        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        kernel_d = kernel_h = kernel_w = self.kernel_size
        in_channels = self.in_channels
        out_channels = self.out_channels

        output = self.conv_transpose3d_cuda.conv_transpose3d_cuda(
            x,
            self.weight,
            batch_size,
            in_channels,
            out_channels,
            kernel_d,
            kernel_h,
            kernel_w,
            self.stride,
            self.padding,
            self.output_padding,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.groups
        )

        return output

# Test inputs (must be outside the class)
batch_size = 8
in_channels = 48
out_channels = 48
kernel_size = 3
depth = 64
height = 64
width = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]