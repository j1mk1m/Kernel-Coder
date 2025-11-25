import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for LogSumExp followed by ReLU
logsumexp_relu_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void logsumexp_relu_kernel(
    const float* input,
    float* output,
    int B, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D * H * W)
        return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int b = idx / (D * H * W);

    float max_val = -INFINITY;
    float sum = 0.0f;

    for (int c = 0; c < C; ++c) {
        int input_offset = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
        float current_val = input[input_offset];

        if (current_val > max_val) {
            max_val = current_val;
        }

        float exp_val = expf(current_val - max_val);
        sum += exp_val;
    }

    float log_sum = max_val + logf(sum);
    float relu_val = log_sum > 0 ? log_sum : 0.0f;

    int output_offset = b * D * H * W + d * H * W + h * W + w;
    output[output_offset] = relu_val;
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::zeros({B, 1, D, H, W}, input.options());

    const int threads = 256;
    int blocks = (B * D * H * W + threads - 1) / threads;

    logsumexp_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D, H, W);

    return output;
}
"""

logsumexp_relu_header = """
torch::Tensor logsumexp_relu_cuda(torch::Tensor input);
"""

logsumexp_relu = load_inline(
    name="logsumexp_relu",
    cpp_sources=logsumexp_relu_header,
    cuda_sources=logsumexp_relu_source,
    functions=["logsumexp_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.logsumexp_relu = logsumexp_relu  # Custom CUDA kernel

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.logsumexp_relu.logsumexp_relu_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]