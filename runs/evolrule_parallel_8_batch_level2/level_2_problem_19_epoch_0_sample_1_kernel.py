import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom GELU kernel
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define M_SQRT_2_OVER_PI 0.7978845608

__global__ void gelu_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float inner = M_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
        float tanh_val = tanh(inner);
        output[idx] = 0.5 * x * (1.0 + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_forward<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor input);"
)

gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
)

# Custom GroupNorm kernel
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void group_norm_forward(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* weight,
    const scalar_t* bias,
    int N, int C, int H, int W,
    int G, float eps) {

    const int channels_per_group = C / G;
    const int group_size = N * channels_per_group * H * W;
    const int group_offset = blockIdx.x * channels_per_group;

    extern __shared__ float shared[];
    float* sum = shared;
    float* sum_sq = shared + blockDim.x;

    sum[threadIdx.x] = 0.0f;
    sum_sq[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int n = idx / (channels_per_group * H * W);
        int c = (idx / (H * W)) % channels_per_group;
        int h = (idx / W) % H;
        int w = idx % W;

        int input_idx = n * C * H * W + (group_offset + c) * H * W + h * W + w;
        scalar_t val = input[input_idx];
        sum[threadIdx.x] += val;
        sum_sq[threadIdx.x] += val * val;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum[threadIdx.x] += sum[threadIdx.x + s];
            sum_sq[threadIdx.x] += sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = sum[0] / group_size;
        float var = sum_sq[0] / group_size - mean * mean;
        var = 1.0f / sqrtf(var + eps);

        for (int idx = 0; idx < group_size; ++idx) {
            int n = idx / (channels_per_group * H * W);
            int c = (idx / (H * W)) % channels_per_group;
            int h = (idx / W) % H;
            int w = idx % W;

            int input_idx = n * C * H * W + (group_offset + c) * H * W + h * W + w;
            output[input_idx] = (input[input_idx] - mean) * var * weight[blockIdx.x] + bias[blockIdx.x];
        }
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int G,
    float eps = 1e-5) {

    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = G;

    dim3 grid(blocks);
    dim3 block(threads);
    int shared_size = 2 * threads * sizeof(float);

    group_norm_forward<float><<<grid, block, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W,
        G, eps
    );

    return output;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int G, float eps = 1e-5);"
)

group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.gelu = gelu
        self.group_norm = group_norm
        self.weight = nn.Parameter(torch.ones(num_groups))
        self.bias = nn.Parameter(torch.zeros(num_groups))
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.gelu.gelu_cuda(x)
        x = self.group_norm.group_norm_cuda(
            x, self.weight, self.bias, self.num_groups
        )
        return x