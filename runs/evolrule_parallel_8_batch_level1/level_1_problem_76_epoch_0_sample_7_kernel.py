import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel code
conv1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* weight, float* output,
                             int batch_size, int in_channels, int out_channels,
                             int length, int output_length,
                             int kernel_size, int stride, int dilation,
                             const float* bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int n = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int oc = rem / output_length;
    int t = rem % output_length;

    float acc = 0.0f;
    int start = t * stride;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int pos = start + k * dilation;
            int in_offset = n * in_channels * length + ic * length + pos;
            float in_val = input[in_offset];

            int w_offset = oc * in_channels * kernel_size + ic * kernel_size + k;
            float w_val = weight[w_offset];

            acc += in_val * w_val;
        }
    }

    if (bias) {
        acc += bias[oc];
    }

    int out_offset = n * out_channels * output_length + oc * output_length + t;
    output[out_offset] = acc;
}

torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         int stride, int dilation, int kernel_size) {
    // Check inputs
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int length = input.size(2);
    int out_channels = weight.size(0);
    int output_length = (length - dilation * (kernel_size -1) - 1) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, out_channels, output_length}, options);

    int threads_per_block = 256;
    int total_elements = batch_size * out_channels * output_length;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_cuda", ([&] {
        conv1d_kernel<<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            length, output_length,
            kernel_size, stride, dilation,
            bias.defined() ? bias.data_ptr<float>() : nullptr
        );
    }));

    return output;
}
"""

conv1d_header = """
torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                         int stride, int dilation, int kernel_size);
"""

# Load the CUDA kernel
conv1d_cuda = load_inline(
    name="conv1d_cuda",
    cpp_sources=conv1d_header,
    cuda_sources=conv1d_source,
    functions=["conv1d_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Initialize weight and bias like PyTorch's Conv1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare bias tensor (empty if None)
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return conv1d_cuda.conv1d_cuda(x, self.weight, bias_tensor,
                                      self.stride, self.dilation, self.kernel_size)