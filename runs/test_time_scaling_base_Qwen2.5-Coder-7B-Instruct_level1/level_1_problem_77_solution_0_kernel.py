import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Helper function to perform the actual convolution
__device__ void conv_transpose3d_device(float* out, const float* inp, const float* weight, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size, int stride, int padding, int dilation) {
    int b = blockIdx.z;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (o >= out_channels || i >= in_channels || b >= batch_size) return;

    float sum = 0.0f;
    int d_out = (b * stride + i * dilation - padding);
    int h_out = (b * stride + o * dilation - padding);
    int w_out = (b * stride + i * dilation - padding);

    for (int k = 0; k < kernel_size; ++k) {
        int d_in = d_out + k;
        int h_in = h_out + k;
        int w_in = w_out + k;

        if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            int in_idx = b * in_channels * depth * height * width + i * depth * height * width + d_in * height * width + h_in * width + w_in;
            int out_idx = b * out_channels * depth * height * width + o * depth * height * width + d_out * height * width + h_out * width + w_out;
            int weight_idx = o * in_channels * kernel_size * kernel_size * kernel_size + i * kernel_size * kernel_size * kernel_size + k * kernel_size * kernel_size + k * kernel_size + k;

            sum += inp[in_idx] * weight[weight_idx];
        }
    }

    out[out_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor inp, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size, int stride, int padding, int dilation) {
    auto out = torch::zeros({batch_size, out_channels, depth, height, width}, torch::dtype(inp.dtype()).device(inp.device()));

    dim3 threads(16, 16, 1);
    dim3 blocks((in_channels + threads.x - 1) / threads.x, (out_channels + threads.y - 1) / threads.y, batch_size);

    conv_transpose3d_device<<<blocks, threads>>>(out.data_ptr<float>(), inp.data_ptr<float>(), weight.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding, dilation);

    return out;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor inp, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_cuda(x, self.weight, x.size(0), x.size(1), self.weight.size(0), x.size(2), x.size(3), x.size(4), self.weight.size(2), stride, padding, dilation) + self.bias.view(1, -1, 1, 1, 1) if self.bias is not None else conv_transpose3d_cuda(x, self.weight, x.size(0), x.size(1), self.weight.size(0), x.size(2), x.size(3), x.size(4), self.weight.size(2), stride, padding, dilation)


# Test code
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation)
inputs = get_inputs()
output_new = model_new(inputs[0])
print(output_new.shape)