import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void pointwise_convolution_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    int B,
    int C_in,
    int C_out,
    int H,
    int W) {
    int b = blockIdx.x / (H * W);
    int rem = blockIdx.x % (H * W);
    int h = rem / W;
    int w = rem % W;

    int c = threadIdx.x;

    if (c >= C_out) return;

    extern __shared__ scalar_t shared_weight[];

    // Load the weight matrix into shared memory
    for (int c_in = 0; c_in < C_in; ++c_in) {
        shared_weight[c * C_in + c_in] = weight[c * C_in + c_in];
    }
    __syncthreads();

    scalar_t sum = 0.0;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        int input_offset = b * C_in * H * W + c_in * H * W + h * W + w;
        sum += input[input_offset] * shared_weight[c * C_in + c_in];
    }

    if (bias) {
        sum += bias[c];
    }

    int output_offset = b * C_out * H * W + c * H * W + h * W + w;
    output[output_offset] = sum;
}

torch::Tensor pointwise_convolution_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    input = input.contiguous();
    weight = weight.contiguous();

    int B = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);

    auto output = torch::empty({B, C_out, H, W}, input.options());

    dim3 threads(C_out);
    dim3 blocks(B * H * W);

    auto bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_convolution_cuda", ([&]{
        using scalar_t = scalar_t;
        size_t smem_size = C_out * C_in * sizeof(scalar_t);

        pointwise_convolution_kernel<scalar_t><<<blocks, threads, smem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_ptr,
            output.data_ptr<scalar_t>(),
            B, C_in, C_out, H, W);
    }));

    return output;
}
"""

pointwise_convolution = load_inline(
    name="pointwise_convolution",
    cuda_sources=cuda_source,
    functions=["pointwise_convolution_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return pointwise_convolution.pointwise_convolution_cuda(x, self.weight, self.bias)
        else:
            return pointwise_convolution.pointwise_convolution_cuda(x, self.weight, torch.empty(0, device=x.device, dtype=x.dtype))