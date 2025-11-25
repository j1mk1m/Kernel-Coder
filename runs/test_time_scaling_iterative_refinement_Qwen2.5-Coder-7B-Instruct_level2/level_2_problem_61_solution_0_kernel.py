import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void conv_transpose_3d_forward_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w) {
  
  CUDA_KERNEL_LOOP(index, batch_size * out_channels * D_out * H_out * W_out) {
    int d_o = index / (H_out * W_out);
    int h_o = (index % (H_out * W_out)) / W_out;
    int w_o = index % W_out;

    int d_i = d_o * stride_d - padding_d;
    int h_i = h_o * stride_h - padding_h;
    int w_i = w_o * stride_w - padding_w;

    int channel_index = 0;
    for (int c = 0; c < in_channels; ++c) {
      int d_s = 0;
      for (int dd = 0; dd <= D_in - 1; ++dd) {
        if (d_i + dd >= 0 && d_i + dd < D_in) {
          int h_s = 0;
          for (int hh = 0; hh <= H_in - 1; ++hh) {
            if (h_i + hh >= 0 && h_i + hh < H_in) {
              int w_s = 0;
              for (int ww = 0; ww <= W_in - 1; ++ww) {
                if (w_i + ww >= 0 && w_i + ww < W_in) {
                  int input_index = ((c * D_in + dd) * H_in + hh) * W_in + ww;
                  int weight_index = ((channel_index * D_in + dd) * H_in + hh) * W_in + ww;
                  output[index] += input[input_index] * weight[weight_index];
                  w_s++;
                }
              }
              h_s++;
            }
          }
          d_s++;
        }
      }
      channel_index++;
    }
  }
}

torch::Tensor conv_transpose_3d_forward_cuda(
    torch::Tensor input, torch::Tensor weight, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w) {
  
  auto batch_size = input.size(0);
  auto in_channels = input.size(1);
  auto out_channels = weight.size(0);
  auto D_in = input.size(2);
  auto H_in = input.size(3);
  auto W_in = input.size(4);
  auto D_out = (D_in - 1) * stride_d + 1 - 2 * padding_d;
  auto H_out = (H_in - 1) * stride_h + 1 - 2 * padding_h;
  auto W_out = (W_in - 1) * stride_w + 1 - 2 * padding_w;

  auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());

  const int block_size = 256;
  const int num_blocks = (output.numel() + block_size - 1) / block_size;

  conv_transpose_3d_forward_kernel<<<num_blocks, block_size>>>(
      input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
      batch_size, in_channels, out_channels, D_in, H_in, W_in, D_out, H_out, W_out, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w);

  return output;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w);"
)

# Compile the inline CUDA code for transposed 3D convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_forward_cuda(x, self.weight, stride_d=1, stride_h=1, stride_w=1, padding_d=0, padding_h=0, padding_w=0)
        x = self.relu(x)
        x = self.group_norm(x)
        return x