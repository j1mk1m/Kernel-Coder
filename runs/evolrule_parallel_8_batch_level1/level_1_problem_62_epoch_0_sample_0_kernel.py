import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_height * out_width) {
        return;
    }

    int b = idx / (out_channels * out_height * out_width);
    int remainder = idx % (out_channels * out_height * out_width);
    int oc = remainder / (out_height * out_width);
    int yx = remainder % (out_height * out_width);
    int y = yx / out_width;
    int x = yx % out_width;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_y = y + kh;
                int input_x = x + kw;
                int input_offset = b * in_channels * input_height * input_width +
                                   ic * input_height * input_width +
                                   input_y * input_width + input_x;
                float in_val = input[input_offset];

                int weight_offset = oc * in_channels * kernel_h * kernel_w +
                                    ic * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                float wt_val = weight[weight_offset];

                sum += in_val * wt_val;
            }
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }

    int output_offset = b * out_channels * out_height * out_width +
                        oc * out_height * out_width +
                        y * out_width + x;
    output[output_offset] = sum;
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    bool has_bias
) {
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    int total_threads = batch_size * out_channels * out_height * out_width;
    const int threadsPerBlock = 256;
    const int blocks = (total_threads + threadsPerBlock -1) / threadsPerBlock;
    
    custom_conv2d_kernel<<<blocks, threadsPerBlock>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        out_height,
        out_width,
        has_bias
    );
    
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    bool has_bias
);
"""

custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters like PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch_size = x.size(0)
        input_height = x.size(2)
        input_width = x.size(3)
        kernel_h, kernel_w = self.kernel_size
        
        # Compute output dimensions
        out_height = (input_height + 2 * self.padding - self.dilation * (kernel_h - 1) - 1) // self.stride + 1
        out_width = (input_width + 2 * self.padding - self.dilation * (kernel_w - 1) - 1) // self.stride + 1
        
        # Prepare bias tensor
        has_bias = self.bias is not None
        bias_tensor = self.bias if has_bias else torch.tensor([0.0], device=x.device)
        
        # Call the CUDA function with parameters
        output = custom_conv2d.custom_conv2d_cuda(
            x,
            self.weight,
            bias_tensor,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_height,
            input_width,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            has_bias
        )
        
        return output