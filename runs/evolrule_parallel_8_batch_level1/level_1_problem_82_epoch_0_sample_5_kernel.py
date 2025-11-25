import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width) {

    int batch = blockIdx.x;
    int channel = blockIdx.y;
    int block_z = blockIdx.z;

    // Compute block_h and block_w from block_z
    int num_w_blocks = (output_width + blockDim.y - 1) / blockDim.y;
    int block_h = block_z / num_w_blocks;
    int block_w = block_z % num_w_blocks;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute spatial indices
    int h_out = block_h * blockDim.x + tx;
    int w_out = block_w * blockDim.y + ty;

    if (h_out >= output_height || w_out >= output_width) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int input_h = h_out * stride + i - padding;
            int input_w = w_out * stride + j - padding;

            // Check if input indices are within bounds (due to padding)
            if (input_h < 0 || input_h >= input_height || input_w < 0 || input_w >= input_width) {
                continue;
            }

            // Calculate input's linear index
            int input_offset = batch * in_channels * input_height * input_width
                + channel * input_height * input_width
                + input_h * input_width + input_w;

            // Calculate weight's linear index
            int weight_offset = channel * kernel_size * kernel_size
                + i * kernel_size + j;

            sum += input[input_offset] * weight[weight_offset];
        }
    }

    // Calculate output's linear index
    int output_offset = batch * in_channels * output_height * output_width
        + channel * output_height * output_width
        + h_out * output_width + w_out;

    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                   int kernel_size, int stride, int padding) {
    auto device = input.device();
    TORCH_CHECK(weight.device() == device, "Input and weight must be on the same device");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_size + 2 * padding) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, output_height, output_width}, 
                              torch::device(device).dtype(torch::kFloat32));

    const int block_dim_x = 32;
    const int block_dim_y = 32;
    dim3 block(block_dim_x, block_dim_y, 1);

    int num_h_blocks = (output_height + block_dim_x - 1) / block_dim_x;
    int num_w_blocks = (output_width + block_dim_y - 1) / block_dim_y;
    int grid_dim_z = num_h_blocks * num_w_blocks;

    dim3 grid(batch_size, in_channels, grid_dim_z);

    depthwise_conv2d_forward<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, input_height, input_width,
        kernel_size, stride, padding, output_height, output_width
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cuda_sources=depthwise_conv_source,
    cpp_sources=depthwise_conv_cpp_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        
        self.depthwise_conv = depthwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.depthwise_conv.depthwise_conv2d_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output