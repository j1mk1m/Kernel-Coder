import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, float* output, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z * blockDim.z + threadIdx.z;
    if (oc >= out_channels || ic >= in_channels) {
        return;
    }

    int o_depth = depth - kernel_size + 1;
    int o_height = height - kernel_size + 1;
    int o_width = width - kernel_size + 1;

    for (int od = 0; od < o_depth; ++od) {
        for (int oh = 0; oh < o_height; ++oh) {
            for (int ow = 0; ow < o_width; ++ow) {
                float sum = 0.0f;
                for (int kd = 0; kd < kernel_size; ++kd) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int id = od + kd;
                            int ih = oh + kh;
                            int iw = ow + kw;
                            sum += input[id * in_channels * height * width + ic * height * width + ih * width + iw] * weight[oc * in_channels * kernel_size * kernel_size * kernel_size + ic * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw];
                        }
                    }
                }
                output[oc * o_depth * o_height * o_width + od * o_height * o_width + oh * o_width + ow] = sum;
            }
        }
    }
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({out_channels, depth, height, width}, input.options());

    const int block_size = 8;
    dim3 grid(out_channels / block_size, in_channels / block_size, 1);
    dim3 block(block_size, block_size, 1);

    conv3d_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), in_channels, out_channels, depth, height, width, kernel_size);

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = conv3d
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x, self.weight)
        x = self.norm(x)
        x = torch.min(x, torch.tensor(min_value, device=x.device))
        x = torch.clamp(x, min=min_value, max=max_value)
        x = self.dropout(x)
        return x