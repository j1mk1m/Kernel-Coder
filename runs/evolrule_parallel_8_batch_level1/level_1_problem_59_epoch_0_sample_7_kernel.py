import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv3d_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor custom_conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    // Reshape input to (N*D, C_in, H, W)
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int D = input.size(4);
    
    // Permute to (N, D, C_in, H, W) then reshape to (N*D, C_in, H, W)
    auto input_reshaped = input.permute({0, 4, 1, 2, 3}).contiguous().view({N * D, C_in, H, W});

    // Reshape weight to remove depth dimension (since kernel_size[2] is 1)
    auto weight_reshaped = weight.view({weight.size(0), weight.size(1), weight.size(2), weight.size(3)});
    
    // Apply 2D convolution on the reshaped input
    auto output = torch::nn::functional::conv2d(
        input_reshaped,
        weight_reshaped,
        bias,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        groups
    );
    
    // Reshape back to original dimensions
    int C_out = output.size(1);
    int H_out = output.size(2);
    int W_out = output.size(3);
    
    output = output.view({N, D, C_out, H_out, W_out});
    output = output.permute({0, 2, 3, 4, 1});
    
    return output;
}
"""

custom_conv3d_cpp_source = (
    "torch::Tensor custom_conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups);"
)

custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources=custom_conv3d_cpp_source,
    cuda_sources=custom_conv3d_source,
    functions=["custom_conv3d_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias = self.bias if self.bias is not None else torch.tensor([])
        return custom_conv3d.custom_conv3d_forward(
            x,
            self.weight,
            bias,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.groups
        )