import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel implementation for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, float* output,
                             int batch_size, int in_channels, int out_channels,
                             int input_depth, int input_height, int input_width,
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation,
                             int output_depth, int output_height, int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width)
        return;

    int batch = idx / (out_channels * output_depth * output_height * output_width);
    int remainder = idx % (out_channels * output_depth * output_height * output_width);
    int oc = remainder / (output_depth * output_height * output_width);
    remainder %= (output_depth * output_height * output_width);
    int d_out = remainder / (output_height * output_width);
    remainder %= (output_height * output_width);
    int h_out = remainder / output_width;
    int w_out = remainder % output_width;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_d = d_out * stride + kd * dilation - padding;
                    int input_h = h_out * stride + kh * dilation - padding;
                    int input_w = w_out * stride + kw * dilation - padding;

                    if (input_d >= 0 && input_d < input_depth &&
                        input_h >= 0 && input_h < input_height &&
                        input_w >= 0 && input_w < input_width) {
                        int input_offset = batch * in_channels * input_depth * input_height * input_width
                                         + ic * input_depth * input_height * input_width
                                         + input_d * input_height * input_width
                                         + input_h * input_width + input_w;
                        float input_val = input[input_offset];

                        int weight_offset = oc * in_channels * kernel_size * kernel_size * kernel_size
                                          + ic * kernel_size * kernel_size * kernel_size
                                          + kd * kernel_size * kernel_size
                                          + kh * kernel_size + kw;
                        float weight_val = weight[weight_offset];

                        sum += input_val * weight_val;
                    }
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    auto device = input.device();
    if (input.device() != weight.device()) {
        AT_ERROR("Input and weight must be on the same device");
    }
    if (!input.is_contiguous()) {
        AT_ERROR("Input tensor must be contiguous");
    }
    if (!weight.is_contiguous()) {
        AT_ERROR("Weight tensor must be contiguous");
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int in_channels_weight = weight.size(1);
    if (in_channels != in_channels_weight) {
        AT_ERROR("Input channels must match weight's input channels");
    }

    int output_depth = (input_depth + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;

    auto options = input.options();
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, options);

    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;

    const int threadsPerBlock = 256;
    dim3 blocks((total_elements + threadsPerBlock - 1) / threadsPerBlock);
    dim3 threads(threadsPerBlock);

    conv3d_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                      weight.data_ptr<float>(),
                                      output.data_ptr<float>(),
                                      batch_size,
                                      in_channels,
                                      out_channels,
                                      input_depth,
                                      input_height,
                                      input_width,
                                      kernel_size,
                                      stride,
                                      padding,
                                      dilation,
                                      output_depth,
                                      output_height,
                                      output_width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);
"""

# Compile the CUDA extension
conv3d = load_inline(
    name="conv3d",
    cuda_sources=conv3d_source,
    cpp_sources=conv3d_cpp_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        if groups != 1:
            raise NotImplementedError("Groups > 1 are not supported in this optimized implementation")

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.bias_param = None

    def forward(self, x):
        output = conv3d.conv3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)

        if self.bias_param is not None:
            output = output + self.bias_param.view(1, -1, 1, 1, 1)

        return output