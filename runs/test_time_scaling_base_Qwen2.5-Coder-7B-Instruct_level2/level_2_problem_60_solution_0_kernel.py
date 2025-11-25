import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Load the custom CUDA kernels
convtranspose3d_module = load_inline(
    name="convtranspose3d",
    cpp_sources="convtranspose3d.cpp",
    cuda_sources="convtranspose3d.cu",
    functions=["convtranspose3d_backward_cuda"],
    verbose=True,
)

swish_module = load_inline(
    name="swish",
    cpp_sources="swish.cpp",
    cuda_sources="swish.cu",
    functions=["swish_forward_cuda", "swish_backward_cuda"],
    verbose=True,
)

groupnorm_module = load_inline(
    name="groupnorm",
    cpp_sources="groupnorm.cpp",
    cuda_sources="groupnorm.cu",
    functions=["groupnorm_forward_cuda", "groupnorm_backward_cuda"],
    verbose=True,
)

hardswish_module = load_inline(
    name="hardswish",
    cpp_sources="hardswish.cpp",
    cuda_sources="hardswish.cu",
    functions=["hardswish_forward_cuda", "hardswish_backward_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = swish_module.swish_forward_cuda(x)
        x = self.group_norm(x)
        x = hardswish_module.hardswish_forward_cuda(x)
        return x


# Example usage
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, groups, eps)
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)