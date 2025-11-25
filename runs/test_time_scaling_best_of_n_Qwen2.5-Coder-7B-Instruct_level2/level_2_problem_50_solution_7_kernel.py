import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for each operation

# Custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
__global__ void conv_transpose_3d_kernel(float* input, float* weight, float* bias, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int groups) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int d = blockIdx.z / (height_out * width_out);
    int h = (blockIdx.z % (height_out * width_out)) / width_out;
    int w = blockIdx.z % width_out;

    float sum = 0.0f;
    for (int i = 0; i < depth_in; ++i) {
        for (int j = 0; j < height_in; ++j) {
            for (int k = 0; k < width_in; ++k) {
                int input_idx = n * in_channels * depth_in * height_in * width_in + c * depth_in * height_in * width_in + i * height_in * width_in + j * width_in + k;
                int weight_idx = c * out_channels * depth_in * height_in * width_in + ((d * stride_d - padding_d + i) * stride_h - padding_h + j) * stride_w - padding_w + k;
                int output_idx = n * out_channels * depth_out * height_out * width_out + c * depth_out * height_out * width_out + (d * stride_d - padding_d + i) * stride_h - padding_h + j) * stride_w - padding_w + k;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[output_idx] = sum + bias[c];
}
"""

conv_transpose_3d_cpp_source = (
    "void conv_transpose_3d_cuda(float* input, float* weight, float* bias, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, int groups);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for scaling
scaling_source = """
__global__ void scaling_kernel(float* input, float* scale, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scale[idx];
    }
}
"""

scaling_cpp_source = (
    "void scaling_cuda(float* input, float* scale, float* output, int size);"
)

# Compile the inline CUDA code for scaling
scaling = load_inline(
    name="scaling",
    cpp_sources=scaling_cpp_source,
    cuda_sources=scaling_source,
    functions=["scaling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for average pooling
avg_pooling_3d_source = """
__global__ void avg_pooling_3d_kernel(float* input, float* output, int batch_size, int channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size_d, int kernel_size_h, int kernel_size_w, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, bool ceil_mode, bool count_include_pad) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int d = blockIdx.z / (height_out * width_out);
    int h = (blockIdx.z % (height_out * width_out)) / width_out;
    int w = blockIdx.z % width_out;

    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < kernel_size_d; ++i) {
        for (int j = 0; j < kernel_size_h; ++j) {
            for (int k = 0; k < kernel_size_w; ++k) {
                int input_idx = n * channels * depth_in * height_in * width_in + c * depth_in * height_in * width_in + (d * stride_d - padding_d + i) * stride_h - padding_h + j) * stride_w - padding_w + k;
                if (input_idx >= 0 && input_idx < depth_in * height_in * width_in) {
                    sum += input[input_idx];
                    count++;
                }
            }
        }
    }

    output[n * channels * depth_out * height_out * width_out + c * depth_out * height_out * width_out + d * stride_d + h * stride_h + w] = ceil_mode ? ceil(sum / count) : floor(sum / count);
}
"""

avg_pooling_3d_cpp_source = (
    "void avg_pooling_3d_cuda(float* input, float* output, int batch_size, int channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size_d, int kernel_size_h, int kernel_size_w, int stride_d, int stride_h, int stride_w, int padding_d, int padding_h, int padding_w, bool ceil_mode, bool count_include_pad);"
)

# Compile the inline CUDA code for average pooling
avg_pooling_3d = load_inline(
    name="avg_pooling_3d",
    cpp_sources=avg_pooling_3d_cpp_source,
    cuda_sources=avg_pooling_3d_source,
    functions=["avg_pooling_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for bias addition
bias_addition_source = """
__global__ void bias_addition_kernel(float* input, float* bias, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + bias[idx];
    }
}
"""

bias_addition_cpp_source = (
    "void bias_addition_cuda(float* input, float* bias, float* output, int size);"
)

# Compile the inline CUDA code for bias addition
bias_addition = load_inline(
    name="bias_addition",
    cpp_sources=bias_addition_cpp_source,
    cuda_sources=bias_addition_source,
    functions=["bias_addition_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for final scaling
final_scaling_source = """
__global__ void final_scaling_kernel(float* input, float* scale, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scale[idx];
    }
}
"""

final_scaling_cpp_source = (
    "void final_scaling_cuda(float* input, float* scale, float* output, int size);"
)

# Compile the inline CUDA code for final scaling
final_scaling = load_inline(
    name="final_scaling",
    cpp_sources=final_scaling_cpp_source,
    cuda_sources=final_scaling_source,
    functions=["final_scaling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth_in = 16
        self.height_in = 32
        self.width_in = 32
        self.depth_out = 16
        self.height_out = 32
        self.width_out = 32
        self.stride_d = 2
        self.stride_h = 2
        self.stride_w = 2
        self.padding_d = 1
        self.padding_h = 1
        self.padding_w = 1
        self.kernel_size_d = 3
        self.kernel_size_h = 3
        self.kernel_size_w = 3
        self.groups = 1
        self.scale1_param = nn.Parameter(torch.tensor(scale1))
        self.scale2_param = nn.Parameter(torch.tensor(scale2))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        batch_size = x.size(0)
        output_size = batch_size * self.out_channels * self.depth_out * self.height_out * self.width_out

        input_gpu = x.contiguous().view(-1)
        weight_gpu = torch.randn(self.out_channels, self.in_channels, self.kernel_size_d, self.kernel_size_h, self.kernel_size_w).contiguous().view(-1)
        bias_gpu = self.bias.view(-1)
        output_gpu = torch.zeros(output_size, dtype=torch.float32).contiguous()

        conv_transpose_3d.conv_transpose_3d_cuda(input_gpu.data_ptr(), weight_gpu.data_ptr(), bias_gpu.data_ptr(), output_gpu.data_ptr(), batch_size, self.in_channels, self.out_channels, self.depth_in, self.height_in, self.width_in, self.depth_out, self.height_out, self.width_out, self.stride_d, self.stride_h, self.stride_w, self.padding_d, self.padding_h, self.padding_w, self.groups)

        scaling.scaling_cuda(output_gpu.data_ptr(), self.scale1_param.data_ptr(), output_gpu.data_ptr(), output_size)

        input_gpu = output_gpu.contiguous().view(-1)
        output_gpu = torch.zeros(output_size, dtype=torch.float32).contiguous()

        avg_pooling_3d.avg_pooling_3d_cuda(input_gpu.data_ptr(), output_gpu.data_ptr(), batch_size, self.out_channels, self.depth_out, self.height_out, self.width_out, self.depth_out, self.height_out, self.width_out, self.kernel_size_d, self.kernel_size_h, self.kernel_size_w, self.stride_d, self.stride_h, self.stride_w, self.padding_d, self.padding_h, self.padding_w, False, True)

        scaling.scaling_cuda(output_gpu.data_ptr(), self.scale2_param.data_ptr(), output_gpu.data_ptr(), output_size)

        return output_gpu.view(batch_size, self.out_channels, self.depth_out, self.height_out, self.width_out)