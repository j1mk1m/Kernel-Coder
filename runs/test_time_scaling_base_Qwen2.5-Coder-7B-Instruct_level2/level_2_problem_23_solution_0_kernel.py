import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
convolution_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int D, int H, int W, int kernel_size) {
    int b = blockIdx.x / (H * W);
    int h = (blockIdx.x / W) % H;
    int w = blockIdx.x % W;
    int c_out = blockIdx.y;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int d = 0; d < kernel_size; ++d) {
            for (int dh = 0; dh < kernel_size; ++dh) {
                for (int dw = 0; dw < kernel_size; ++dw) {
                    int id = b * in_channels * D * H * W + c_in * D * H * W + (h - dh + D / 2) * H * W + (w - dw + W / 2) * W + d;
                    int iw = b * out_channels * D * H * W + c_out * D * H * W + h * H * W + w * W + d;
                    sum += input[id] * weight[iw];
                }
            }
        }
    }
    output[b * out_channels * D * H * W + c_out * D * H * W + h * H * W + w * W] = sum;
}

torch::Tensor convolution_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels}, input.options());

    const int block_size = 16;
    const int grid_size = (D * H * W + block_size - 1) / block_size;

    convolution_3d_kernel<<<grid_size * out_channels, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, D, H, W, kernel_size);

    return output;
}
"""

convolution_3d_cpp_source = (
    "torch::Tensor convolution_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
convolution_3d = load_inline(
    name="convolution_3d",
    cpp_sources=convolution_3d_cpp_source,
    cuda_sources=convolution_3d_source,
    functions=["convolution_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Group Normalization
group_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_normalization_kernel(const float* input, float* output, float* mean, float* inv_std, int batch_size, int in_channels, int D, int H, int W, int num_groups) {
    int g = blockIdx.x / (H * W);
    int h = (blockIdx.x / W) % H;
    int w = blockIdx.x % W;
    int c_in = blockIdx.y;

    float sum = 0.0f;
    float sum_sqr = 0.0f;
    for (int i = 0; i < D; ++i) {
        int id = g * in_channels * D * H * W + c_in * D * H * W + h * H * W + w * W + i;
        sum += input[id];
        sum_sqr += input[id] * input[id];
    }

    mean[g * in_channels + c_in] = sum / (D * H * W);
    inv_std[g * in_channels + c_in] = 1.0f / std::sqrt(sum_sqr / (D * H * W) - mean[g * in_channels + c_in] * mean[g * in_channels + c_in] + 1e-5);

    for (int i = 0; i < D; ++i) {
        int id = g * in_channels * D * H * W + c_in * D * H * W + h * H * W + w * W + i;
        output[id] = (input[id] - mean[g * in_channels + c_in]) * inv_std[g * in_channels + c_in];
    }
}

torch::Tensor group_normalization_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto num_groups = in_channels / 8;

    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({batch_size, in_channels});
    auto inv_std = torch::zeros({batch_size, in_channels});

    const int block_size = 16;
    const int grid_size = (D * H * W + block_size - 1) / block_size;

    group_normalization_kernel<<<grid_size * in_channels, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), inv_std.data_ptr<float>(), batch_size, in_channels, D, H, W, num_groups);

    return output;
}
"""

group_normalization_cpp_source = (
    "torch::Tensor group_normalization_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Group Normalization
group_normalization = load_inline(
    name="group_normalization",
    cpp_sources=group_normalization_cpp_source,
    cuda_sources=group_normalization_source,
    functions=["group_normalization_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = convolution_3d
        self.group_norm = group_normalization

    def forward(self, x):
        x = self.conv.convolution_3d_cuda(x, self.weight)
        x = self.group_norm.group_normalization_cuda(x)
        x = x.mean(dim=[1, 2, 3, 4])
        return x

# Initialize weights for convolution
model = Model(in_channels, out_channels, kernel_size, num_groups)
weight = model.conv.weight.clone().detach().requires_grad_(False)

# Create the new model
model_new = ModelNew(in_channels, out_channels, kernel_size, num_groups)
model_new.weight = weight