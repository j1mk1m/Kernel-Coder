import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for pointwise convolution
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
                                       const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
                                       torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output) {
    const int batch_idx = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int in_channel = threadIdx.x;
    const int h = threadIdx.y;
    const int w = threadIdx.z;

    scalar_t sum = 0;
    for (int c = 0; c < weight.size(1); ++c) {
        sum += input[batch_idx][c][h][w] * weight[out_channel][c][0][0];
    }

    if (in_channel == 0 && h < output.size(2) && w < output.size(3)) {
        output[batch_idx][out_channel][h][w] = sum;
    }
}

std::tuple<torch::Tensor, torch::Tensor> pointwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight) {
    const int batch_size = input.size(0);
    const int out_channels = weight.size(0);
    const int in_channels = weight.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 threads(in_channels, 1, 1);
    dim3 blocks(batch_size, out_channels, 1);

    // Launch kernel using half-warp to handle in_channels dimension
    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>()
        );
    }));

    return std::make_tuple(output, weight);
}
"""

pointwise_conv_cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> pointwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight);
"""

pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-DENABLE_CUDA_FP16" if torch.cuda.is_available() else ""],
    extra_cuda_cflags=["--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Custom CUDA operator binding
        self.pointwise_conv = pointwise_conv

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output, _ = self.pointwise_conv.pointwise_conv2d_cuda(x, self.weight)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output

# Ensure get_init_inputs remains unchanged
def get_init_inputs():
    return [in_channels, out_channels]