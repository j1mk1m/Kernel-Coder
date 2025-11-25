import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int N, int C, int H, int W, int K, int S, int P) {
    int n = blockIdx.x / (H * W);
    int h = (blockIdx.x % (H * W)) / W;
    int w = blockIdx.x % W;
    int c = threadIdx.x;

    if (c < C && n < N) {
        float sum = 0.0f;
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = h * S + kh - P;
                int iw = w * S + kw - P;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    sum += input[n * C * H * W + c * H * W + ih * W + iw] * weight[kh * K * C + kw * C + c];
                }
            }
        }
        output[n * C * H * W + c * H * W + h * W + w] = sum;
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto K = weight.size(2);
    auto S = stride;
    auto P = padding;
    auto OH = (H + 2 * P - K) / S + 1;
    auto OW = (W + 2 * P - K) / S + 1;
    auto out = torch::zeros({N, C, OH, OW}, input.options());

    const int block_size = 256;
    const int num_blocks = (C + block_size - 1) / block_size;

    convolution_kernel<<<N * OH * OW, num_blocks>>>(input.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, K, S, P);

    return out;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for group normalization
group_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_normalization_kernel(const float* input, float* mean, float* var, float* output, int N, int C, int G, int H, int W, float eps) {
    int g = blockIdx.x;
    int nc = blockIdx.y * C;
    int h = blockIdx.z * H;
    int w = blockIdx.w * W;
    int c = threadIdx.x;

    if (g < G && nc + c < C && h < N * H && w < N * W) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < G; ++i) {
            sum += input[(nc + i) * N * H * W + h * W + w];
            sq_sum += input[(nc + i) * N * H * W + h * W + w] * input[(nc + i) * N * H * W + h * W + w];
        }
        mean[g * C + c] = sum / (G * N * H * W);
        var[g * C + c] = sq_sum / (G * N * H * W) - mean[g * C + c] * mean[g * C + c] + eps;
        output[(nc + c) * N * H * W + h * W + w] = (input[(nc + c) * N * H * W + h * W + w] - mean[g * C + c]) / sqrt(var[g * C + c]);
    }
}

torch::Tensor group_normalization_cuda(torch::Tensor input, int groups, float eps) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto G = groups;
    auto H = input.size(2);
    auto W = input.size(3);
    auto OH = H;
    auto OW = W;
    auto out = torch::zeros({N, C, OH, OW}, input.options());
    auto mean = torch::zeros({G, C}, input.options().dtype(torch::kFloat32));
    auto var = torch::zeros({G, C}, input.options().dtype(torch::kFloat32));

    const int block_size = 256;
    const int num_blocks = (C + block_size - 1) / block_size;

    group_normalization_kernel<<<G * OH * OW, num_blocks>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), out.data_ptr<float>(), N, C, G, H, W, eps);

    return out;
}
"""

group_normalization_cpp_source = (
    "torch::Tensor group_normalization_cuda(torch::Tensor input, int groups, float eps);"
)

# Compile the inline CUDA code for group normalization
group_normalization = load_inline(
    name="group_normalization",
    cpp_sources=group_normalization_cpp_source,
    cuda_sources=group_normalization_source,
    functions=["group_normalization_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for tanh
tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanh(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto out = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

tanh_cpp_source = (
    "torch::Tensor tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for tanh
tanh = load_inline(
    name="tanh",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for hardswish
hard_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hard_swish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * max(0.0f, min(6.0f, input[idx] + 3.0f)) / 6.0f;
    }
}

torch::Tensor hard_swish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto out = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hard_swish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

hard_swish_cpp_source = (
    "torch::Tensor hard_swish_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for hardswish
hard_swish = load_inline(
    name="hard_swish",
    cpp_sources=hard_swish_cpp_source,
    cuda_sources=hard_swish_source,
    functions=["hard_swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for residual addition
residual_addition_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void residual_addition_kernel(const float* input, const float* residual, float* output, int N, int C, int H, int W) {
    int n = blockIdx.x / (H * W);
    int h = (blockIdx.x % (H * W)) / W;
    int w = blockIdx.x % W;
    int c = threadIdx.x;

    if (c < C && n < N) {
        output[n * C * H * W + c * H * W + h * W + w] = input[n * C * H * W + c * H * W + h * W + w] + residual[n * C * H * W + c * H * W + h * W + w];
    }
}

torch::Tensor residual_addition_cuda(torch::Tensor input, torch::Tensor residual) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto out = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (C + block_size - 1) / block_size;

    residual_addition_kernel<<<N * H * W, num_blocks>>>(input.data_ptr<float>(), residual.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W);

    return out;
}
"""

residual_addition_cpp_source = (
    "torch::Tensor residual_addition_cuda(torch::Tensor input, torch::Tensor residual);"
)

# Compile the inline CUDA code for residual addition
residual_addition = load_inline(
    name="residual_addition",
    cpp_sources=residual_addition_cpp_source,
    cuda_sources=residual_addition_source,
    functions=["residual_addition_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for logsumexp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void logsumexp_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int n = blockIdx.x / (H * W);
    int h = (blockIdx.x % (H * W)) / W;
    int w = blockIdx.x % W;
    int c = threadIdx.x;

    if (c < C && n < N) {
        float max_val = -INFINITY;
        for (int i = 0; i < C; ++i) {
            if (input[n * C * H * W + i * H * W + h * W + w] > max_val) {
                max_val = input[n * C * H * W + i * H * W + h * W + w];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < C; ++i) {
            sum += exp(input[n * C * H * W + i * H * W + h * W + w] - max_val);
        }
        output[n * C * H * W + c * H * W + h * W + w] = max_val + log(sum);
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto out = torch::zeros({N, C, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (C + block_size - 1) / block_size;

    logsumexp_kernel<<<N * H * W, num_blocks>>>(input.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W);

    return out;
}
"""

logsumexp_cpp_source = (
    "torch::Tensor logsumexp_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for logsumexp
logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.group_norm = group_normalization
        self.tanh = tanh
        self.hard_swish = hard_swish
        self.residual_addition = residual_addition
        self.logsumexp = logsumexp

    def forward(self, x):
        # Convolution
        x_conv = self.conv.convolution_cuda(x, self.weight, self.stride, self.padding)
        # Group Normalization
        x_norm = self.group_norm.group_normalization_cuda(x_conv, self.groups, self.eps)
        # Tanh
        x_tanh = self.tanh.tanh_cuda(x_norm)
        # HardSwish
        x_hard_swish = self.hard_swish.hard_swish_cuda(x_tanh)
        # Residual Addition
        x_res = self.residual_addition.residual_addition_cuda(x_conv, x_hard_swish)
        # LogSumExp
        x_logsumexp = self.logsumexp.logsumexp_cuda(x_res)
        return x_logsumexp


# Get inputs
inputs = get_inputs()

# Initialize model parameters
model_params = get_init_inputs()

# Create model instance
model = Model(*model_params)

# Create model instance using custom CUDA kernels
model_new = ModelNew(*model_params)

# Forward pass through original model
output_original = model(inputs[0])

# Forward pass through new model using custom CUDA kernels
output_new = model_new(inputs[0])

# Check if outputs match
assert torch.allclose(output_original, output_new, atol=1e-5), "Outputs do not match!"