import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the convolution + scaling CUDA kernel
conv_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_scale_kernel(
    const float* input, const float* weight, float scale,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float val = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                if (h_in < H_in && w_in < W_in) {
                    val += input[
                        n * C_in * H_in * W_in +
                        c_in * H_in * W_in +
                        h_in * W_in +
                        w_in
                    ] * weight[
                        c_out * C_in * K * K +
                        c_in * K * K +
                        kh * K +
                        kw
                    ];
                }
            }
        }
    }
    output[
        n * C_out * H_out * W_out +
        c_out * H_out * W_out +
        h_out * W_out +
        w_out
    ] = val * scale;
}

torch::Tensor conv_scale_cuda(
    torch::Tensor input, torch::Tensor weight, float scale,
    int K, int H_out, int W_out
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    int total = N * C_out * H_out * W_out;
    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;

    conv_scale_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        scale,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K, H_out, W_out
    );

    return output;
}
"""

conv_scale_cpp_source = (
    "torch::Tensor conv_scale_cuda(torch::Tensor input, torch::Tensor weight, float scale, int K, int H_out, int W_out);"
)

# Compile the convolution and scaling kernel
conv_scale = load_inline(
    name="conv_scale",
    cpp_sources=conv_scale_cpp_source,
    cuda_sources=conv_scale_source,
    functions=["conv_scale_cuda"],
    verbose=True
)

# Define the channel-wise minimum CUDA kernel
channel_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_min_kernel(
    const float* input, float* output,
    int N, int C_out, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * W) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int n = idx / (W * H);

    float min_val = FLT_MAX;
    for (int c = 0; c < C_out; ++c) {
        float val = input[
            n * C_out * H * W +
            c * H * W +
            h * W +
            w
        ];
        if (val < min_val) min_val = val;
    }
    output[
        n * H * W + h * W + w
    ] = min_val;
}

torch::Tensor channel_min_cuda(torch::Tensor input) {
    int N = input.size(0);
    int C_out = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::zeros({N, 1, H, W}, input.options());
    int total = N * H * W;
    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;

    channel_min_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_out, H, W
    );

    return output;
}
"""

channel_min_cpp_source = (
    "torch::Tensor channel_min_cuda(torch::Tensor input);"
)

# Compile the channel-wise minimum kernel
channel_min = load_inline(
    name="channel_min",
    cpp_sources=channel_min_cpp_source,
    cuda_sources=channel_min_source,
    functions=["channel_min_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.conv_scale = conv_scale
        self.channel_min = channel_min

    def forward(self, x):
        weight = self.conv.weight
        K = self.conv.kernel_size[0]
        H_in, W_in = x.size(2), x.size(3)
        H_out = H_in - K + 1
        W_out = W_in - K + 1

        conv_scaled = self.conv_scale.conv_scale_cuda(
            x, weight, self.scale_factor, K, H_out, W_out
        )
        result = self.channel_min.channel_min_cuda(conv_scaled)
        return result