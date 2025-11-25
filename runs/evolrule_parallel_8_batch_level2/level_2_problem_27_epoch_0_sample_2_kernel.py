import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom HardSwish CUDA kernel
hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hardswish_kernel(const scalar_t* __restrict__ input,
                                scalar_t* __restrict__ output,
                                const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = input[idx];
        scalar_t res = x * min(max(x + static_cast<scalar_t>(3), static_cast<scalar_t>(0)),
                              static_cast<scalar_t>(6)) / static_cast<scalar_t>(6);
        output[idx] = res;
    }
}

torch::Tensor hardswish_cuda(torch::Tensor input) {
    const int threads = 256;
    const int elements = input.numel();
    const int blocks = (elements + threads - 1) / threads;

    auto output = torch::empty_like(input);
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    scalar_t* output_data = output.data_ptr<scalar_t>();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "hardswish_cuda", ([&] {
        hardswish_kernel<scalar_t><<<blocks, threads>>>(
            input_data, output_data, elements);
    }));

    return output;
}
"""

hardswish = load_inline(
    name="hardswish",
    cpp_sources="",
    cuda_sources=hardswish_source,
    functions=["hardswish_cuda"],
    verbose=True,
    extra_cuda_cflags=["--expt-relaxed-constexpr"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.hardswish = hardswish

    def forward(self, x):
        x = self.conv(x)
        # Replace F.hardswish with custom kernel
        x = self.hardswish.hardswish_cuda(x)
        x = self.group_norm(x)
        x = torch.mean(x, dim=[2, 3, 4])
        return x