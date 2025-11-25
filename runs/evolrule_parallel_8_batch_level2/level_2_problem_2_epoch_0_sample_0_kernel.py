import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 自定义CUDA内核代码
post_processing_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void post_processing_kernel(
    const float* x_data,
    const float* bias_data,
    float scaling_factor,
    float* out_data,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width)
        return;

    // 解析线性索引到空间维度
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (width * height * channels);

    // 获取对应通道的偏置值
    float bias_val = bias_data[c]; // 偏置形状是 (out_channels, 1, 1)

    // 计算流程
    float val = x_data[idx] + bias_val;
    val = fmaxf(fminf(val, 1.0f), 0.0f); // 第一次clamp
    val = val * scaling_factor;
    val = fmaxf(fminf(val, 1.0f), 0.0f); // 第二次clamp
    val = val / scaling_factor;

    out_data[idx] = val;
}

torch::Tensor post_processing_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float scaling_factor
) {
    // 获取输入张量的形状
    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    int total_elements = batch_size * channels * height * width;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    post_processing_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        out.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return out;
}
"""

post_processing_cpp = "torch::Tensor post_processing_cuda(torch::Tensor, torch::Tensor, float);"

# 编译CUDA内核
post_processing = load_inline(
    name="post_processing",
    cpp_sources=post_processing_cpp,
    cuda_sources=post_processing_source,
    functions=["post_processing_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.post_processing = post_processing  # 注册自定义操作

    def forward(self, x):
        x = self.conv_transpose(x)
        # 调用自定义CUDA内核进行后处理
        return self.post_processing.post_processing_cuda(
            x, self.bias, self.scaling_factor
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]