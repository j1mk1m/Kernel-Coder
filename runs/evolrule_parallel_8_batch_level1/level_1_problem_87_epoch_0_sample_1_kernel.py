import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/core/TensorOptions.h>

__global__ void pointwise_conv_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width) {

    int n = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    int o = threadIdx.x;
    if (o >= out_channels) return;

    float sum = 0;
    for (int i = 0; i < in_channels; ++i) {
        int input_idx = n * in_channels * height * width + 
            i * height * width + 
            h * width + w;
        sum += input[input_idx] * weight[o * in_channels + i];
    }

    if (bias) {
        sum += bias[o];
    }

    int output_idx = n * out_channels * height * width + 
        o * height * width + 
        h * width + w;
    output[output_idx] = sum;
}

torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias = c10::nullopt) {
    // Check device and contiguity
    TORCH_CHECK(input.device() == weight.device(), "Input and weight must be on the same device");
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.device() == input.device(), "Bias must be on the same device as input");
    }
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_contiguous(), "Bias tensor must be contiguous");
        TORCH_CHECK(b.size(0) == weight.size(0), "Bias size must match out_channels");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty({batch_size, out_channels, height, width}, 
                              input.options());

    dim3 block(out_channels);
    dim3 grid(batch_size, height, width);

    const float* input_data = input.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    pointwise_conv_kernel<<<grid, block>>>(
        input_data, weight_data, bias_data,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

pointwise_conv_cpp_source = """
#include <torch/extension.h>

torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias = c10::nullopt);
"""

pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_cuda"],
    verbose=True,
    extra_cuda_flags=["-O3"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters using the same method as PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias if self.bias is not None else None
        return pointwise_conv.pointwise_conv_cuda(x, weight, bias)