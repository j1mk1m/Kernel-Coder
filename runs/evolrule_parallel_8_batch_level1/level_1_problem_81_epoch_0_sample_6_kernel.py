import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 定义CUDA内核的源代码
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) return;

    int n = idx / (out_channels * output_height * output_width);
    int c_out = (idx / (output_height * output_width)) % out_channels;
    int h_out = (idx / output_width) % output_height;
    int w_out = idx % output_width;

    float val = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = (h_out + padding - kh * dilation) / stride;
                int w_in = (w_out + padding - kw * dilation) / stride;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int weight_offset = c_out * in_channels * kernel_size * kernel_size
                        + c_in * kernel_size * kernel_size
                        + kh * kernel_size + kw;
                    val += weight[weight_offset] * input[
                        n * in_channels * input_height * input_width
                        + c_in * input_height * input_width
                        + h_in * input_width + w_in
                    ];
                }
            }
        }
    }

    if (bias) {
        val += bias[c_out];
    }

    int output_offset = n * out_channels * output_height * output_width
        + c_out * output_height * output_width
        + h_out * output_width + w_out;
    output[output_offset] = val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    int output_height,
    int output_width
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int input_height = input.size(2);
    int input_width = input.size(3);

    auto options = input.options();
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, options);

    const int threads_per_block = 256;
    int num_elements = batch_size * out_channels * output_height * output_width;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_height,
        input_width,
        output_height,
        output_width,
        stride,
        padding,
        dilation
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    return output;
}
"""

# 定义CUDA内核的头文件
conv_transpose2d_header = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int kernel_size,
    int output_height,
    int output_width
);
"""

# 加载CUDA内核
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cuda_sources=conv_transpose2d_source,
    cpp_sources=conv_transpose2d_header,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

# 新的模型类
class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
        self.cuda_conv = conv_transpose2d  # 将CUDA函数绑定到模型中

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_height = x.size(2)
        input_width = x.size(3)
        # 计算输出尺寸
        output_height = (input_height - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - 2 * self.padding
        output_width = (input_width - 1) * self.stride + self.dilation * (self.kernel_size - 1) + 1 - 2 * self.padding

        # 调用自定义CUDA函数
        return self.cuda_conv.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size,
            output_height,
            output_width
        )