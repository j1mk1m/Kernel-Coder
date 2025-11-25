import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int stride,
    int padding,
    int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * H_out * W_out) return;

    int n = idx / (out_channels * H_out * W_out);
    int rem = idx % (out_channels * H_out * W_out);
    int k = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int i = rem / W_out;
    int j = rem % W_out;

    float y_val = 0.0f;

    int p = kernel_size / 2;

    for (int m = 0; m < in_channels; ++m) {
        for (int s = -p; s <= p; s++) {
            for (int t = -p; t <= p; t++) {
                int x_row = i + s * stride - dilation * p;
                int x_col = j + t * stride - dilation * p;

                if (x_row >= 0 && x_row < H_in && x_col >= 0 && x_col < W_in) {
                    int input_offset = n * in_channels * H_in * W_in +
                                      m * H_in * W_in +
                                      x_row * W_in +
                                      x_col;
                    int s_idx = s + p;
                    int t_idx = t + p;
                    int weight_offset = m * out_channels * kernel_size * kernel_size +
                                       k * kernel_size * kernel_size +
                                       s_idx * kernel_size +
                                       t_idx;
                    y_val += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    int output_offset = n * out_channels * H_out * W_out +
                       k * H_out * W_out +
                       i * W_out +
                       j;
    output[output_offset] = y_val;
}

torch::Tensor conv_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int kernel_size) {
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(1);

    int H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::empty({input.size(0), out_channels, H_out, W_out}, input.options());

    const int num_threads = batch_size * out_channels * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (num_threads + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        in_channels,
        out_channels,
        kernel_size,
        H_in,
        W_in,
        H_out,
        W_out,
        stride,
        padding,
        dilation
    );

    return output;
}
"""

conv_transpose_cpp_source = """
torch::Tensor conv_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int kernel_size);
"""

conv_transpose_cuda = load_inline(
    name="conv_transpose_cuda",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weights with the same method as PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels, kernel_size, kernel_size
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose_cuda.conv_transpose_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size
        )