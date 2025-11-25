import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for pointwise 2D convolution
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void pointwise_conv_kernel(const T* input, const T* weight, const T* bias, T* output,
    int B, int Cin, int Cout, int H, int W) {
    int w = blockIdx.x % W;
    int h = (blockIdx.x / W) % H;
    int batch = blockIdx.x / (H * W);

    int oc = threadIdx.x;

    if (oc >= Cout) return;

    T sum = static_cast<T>(0);
    for (int ic = 0; ic < Cin; ++ic) {
        int input_idx = batch * Cin * H * W + ic * H * W + h * W + w;
        sum += input[input_idx] * weight[oc * Cin + ic];
    }

    if (bias) {
        sum += bias[oc];
    }

    int output_idx = batch * Cout * H * W + oc * H * W + h * W + w;
    output[output_idx] = sum;
}

void pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, torch::Tensor output,
    int B, int Cin, int Cout, int H, int W) {
    const int threads = Cout;
    const int blocks = B * H * W;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "pointwise_conv_cuda", ([&] {
        const T* bias_ptr = bias.has_value() ? bias->data<scalar_t>() : nullptr;
        pointwise_conv_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias_ptr,
            output.data<scalar_t>(),
            B, Cin, Cout, H, W);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
}
"""

pointwise_conv_cpp_source = """
void pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, torch::Tensor output,
    int B, int Cin, int Cout, int H, int W);
"""

# Compile the CUDA kernel
pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Initialize weights and bias like PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Attach the CUDA function
        self.pointwise_conv = pointwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        Cout = self.weight.size(0)
        assert Cin == self.weight.size(1), "Input channels must match weight dimensions"

        output = torch.empty(B, Cout, H, W, device=x.device, dtype=x.dtype)

        # Handle bias parameter
        bias = self.bias if self.bias is not None else None

        # Launch CUDA kernel
        self.pointwise_conv.pointwise_conv_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias,
            output,
            B, Cin, Cout, H, W
        )

        return output