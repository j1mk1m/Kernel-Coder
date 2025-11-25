import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution transpose
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void conv_transpose_forward_kernel(
    const float* input, const float* weight, float* output, int N, int C, int D, int H, int W, int KD, int KH, int KW, int SD, int SH, int SW) {
  CUDA_KERNEL_LOOP(n, N) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int offset_in = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
            float sum = 0.0f;
            for (int kd = 0; kd < KD; ++kd) {
              for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                  int id = d + kd * SD;
                  int ih = h + kh * SH;
                  int iw = w + kw * SW;
                  if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int offset_w = c * KD * KH * KW + kd * KH * KW + kh * KW + kw;
                    sum += input[offset_in] * weight[offset_w];
                  }
                }
              }
            }
            int offset_out = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
            output[offset_out] = sum;
          }
        }
      }
    }
  }
}

torch::Tensor conv_transpose_forward_cuda(
    torch::Tensor input, torch::Tensor weight, int D, int H, int W, int KD, int KH, int KW, int SD, int SH, int SW) {
  auto N = input.size(0);
  auto C = input.size(1);
  auto out_d = (D - 1) * SD + KD;
  auto out_h = (H - 1) * SH + KH;
  auto out_w = (W - 1) * SW + KW;
  auto output = torch::zeros({N, C, out_d, out_h, out_w}, input.options());

  const int block_size = 256;
  const int num_blocks = (N * C * out_d * out_h * out_w + block_size - 1) / block_size;

  conv_transpose_forward_kernel<<<num_blocks, block_size>>>(
      input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C, D, H, W, KD, KH, KW, SD, SH, SW);

  return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_forward_cuda(torch::Tensor input, torch::Tensor weight, int D, int H, int W, int KD, int KH, int KW, int SD, int SH, int SW);"
)

# Compile the inline CUDA code for 3D convolution transpose
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if self.bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_forward_cuda(x, self.weight, x.size(2), x.size(3), x.size(4), self.weight.size(2), self.weight.size(3), self.weight.size(4), self.stride[0], self.stride[1], self.stride[2])
        if self.bias:
            x = x + self.bias.view(1, -1, 1, 1, 1)
        x = self.batch_norm(x)
        x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)  # Subtract mean along spatial dimensions
        return x