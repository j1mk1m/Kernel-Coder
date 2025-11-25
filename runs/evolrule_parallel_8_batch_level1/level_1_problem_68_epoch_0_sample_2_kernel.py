import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from pynvml import *

# Custom CUDA implementation of ConvTranspose3d
conv3d_transpose_src = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void conv_transpose3d_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> output,
    int in_channels, int out_channels, int kernel_depth, int kernel_width,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups) {

  const int B = output.size(0);
  const int D_out = output.size(2);
  const int H_out = output.size(3);
  const int W_out = output.size(4);

  CUDA_1D_KERNEL_LOOP(index, B * D_out * H_out * W_out * out_channels) {
    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int d_out = (index / (W_out * H_out)) % D_out;
    int c_out = (index / (W_out * H_out * D_out)) % out_channels;
    int n = index / (out_channels * D_out * H_out * W_out);

    scalar_t val = 0;
    for (int kd = 0; kd < kernel_depth; ++kd) {
      for (int kh = 0; kh < kernel_width; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
          // Compute input coordinates
          int d_in = (d_out - padding_d - kd * dilation_d) / stride_d;
          int h_in = (h_out - padding_h - kh * dilation_h) / stride_h;
          int w_in = (w_out - padding_w - kw * dilation_w) / stride_w;
          
          if (d_in < 0 || h_in < 0 || w_in < 0 ||
              d_in >= input.size(2) || h_in >= input.size(3) || w_in >= input.size(4)) {
            continue;
          }

          for (int c_in_group = 0; c_in_group < in_channels / groups; ++c_in_group) {
            int c_in = c_in_group + (c_out / (out_channels / groups)) * (in_channels / groups);
            val += input[n][c_in][d_in][h_in][w_in] *
                   weight[c_out][c_in_group][kd][kh][kw];
          }
        }
      }
    }
    output[n][c_out][d_out][h_out][w_out] = val;
  }
}

std::tuple<torch::Tensor, torch::Tensor> conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups) {

  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  const int depth = input.size(2);
  const int height = input.size(3);
  const int width = input.size(4);
  const int out_channels = weight.size(0);
  const int kernel_depth = weight.size(2);
  const int kernel_size = weight.size(3); // assuming kernel_width == kernel_height

  // Compute output shape
  int depth_out = (depth - 1) * stride_d - 2 * padding_d + 
                 dilation_d * (kernel_depth - 1) + output_padding_d + 1;
  int height_out = (height - 1) * stride_h - 2 * padding_h + 
                  dilation_h * (kernel_size - 1) + output_padding_h + 1;
  int width_out = (width - 1) * stride_w - 2 * padding_w + 
                 dilation_w * (kernel_size - 1) + output_padding_w + 1;

  auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, 
                            input.options());

  dim3 blocks(TORCH_EXT_DEFAULT_GRIDDIMS(output.numel()));
  dim3 threads(TORCH_EXT_DEFAULT_BLOCKSIZE);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_forward", ([&] {
    conv_transpose3d_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        in_channels, out_channels, kernel_depth, kernel_size,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        groups);
  }));

  return std::make_tuple(output, weight); // Returning weight for backward
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose3d_forward, "ConvTranspose3d forward");
}
"""

conv3d_transpose_cuda = load(
    name="conv_transpose3d_cuda",
    sources=[conv3d_transpose_src],
    extra_cflags=['-g', '-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'],
    verbose=True
)

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
        self.bias = bias

        # Initialize weights similar to PyTorch's default
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Extract parameters
        kernel_depth, kernel_width, kernel_height = self.kernel_size
        assert kernel_width == kernel_height, "Kernel width and height must be equal"

        # Convert parameters to match CUDA kernel expectations
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding
        dilation_d, dilation_h, dilation_w = (1, 1, 1)  # Assuming default dilation

        # Call custom CUDA forward
        output, _ = conv3d_transpose_cuda.forward(
            x, self.weight, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w,
            dilation_d, dilation_h, dilation_w,
            self.groups
        )

        if self.bias is not None:
            # Add bias if needed (could be fused into kernel later for optimization)
            output = output + self.bias.view(1, -1, 1, 1, 1)

        return output