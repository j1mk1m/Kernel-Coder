import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel and wrapper
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int n,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding,
    int dilation,
    int output_depth,
    int output_height,
    int output_width,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * out_channels * output_depth * output_height * output_width) {
        return;
    }

    int w_out = idx % output_width;
    idx /= output_width;
    int h_out = idx % output_height;
    idx /= output_height;
    int d_out = idx % output_depth;
    idx /= output_depth;
    int c_out = idx % out_channels;
    int n_out = idx / out_channels;

    if (n_out >= n || c_out >= out_channels || d_out >= output_depth ||
        h_out >= output_height || w_out >= output_width) {
        return;
    }

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int input_d = d_out * stride + kd * dilation - padding;
                    int input_h = h_out * stride + kh * dilation - padding;
                    int input_w = w_out * stride + kw * dilation - padding;

                    if (input_d < 0 || input_d >= input_depth ||
                        input_h < 0 || input_h >= input_height ||
                        input_w < 0 || input_w >= input_width) {
                        continue;
                    }

                    int in_offset = n_out * in_channels * input_depth * input_height * input_width +
                                    c_in * input_depth * input_height * input_width +
                                    input_d * input_height * input_width +
                                    input_h * input_width +
                                    input_w;
                    float in_val = input[in_offset];

                    int w_offset = c_out * in_channels * kernel_depth * kernel_height * kernel_width +
                                   c_in * kernel_depth * kernel_height * kernel_width +
                                   kd * kernel_height * kernel_width +
                                   kh * kernel_width +
                                   kw;
                    float w_val = weight[w_offset];

                    acc += in_val * w_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[c_out];
    }

    int out_offset = n_out * out_channels * output_depth * output_height * output_width +
                     c_out * output_depth * output_height * output_width +
                     d_out * output_height * output_width +
                     h_out * output_width +
                     w_out;
    output[out_offset] = acc;
}

torch::Tensor custom_conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    if (!input.is_cuda() || !weight.is_cuda() || (bias.defined() && !bias.is_cuda())) {
        throw std::runtime_error("All tensors must be on CUDA");
    }

    if (input.dtype() != torch::kFloat32 || weight.dtype() != torch::kFloat32) {
        throw std::runtime_error("Tensors must be float32");
    }

    int n = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0);
    int in_channels_weight = weight.size(1);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    if (in_channels != in_channels_weight) {
        throw std::runtime_error("Input channels must match");
    }

    int output_depth = (input_depth + 2*padding - dilation*(kernel_depth-1) - 1) / stride + 1;
    int output_height = (input_height + 2*padding - dilation*(kernel_height-1) - 1) / stride + 1;
    int output_width = (input_width + 2*padding - dilation*(kernel_width-1) - 1) / stride + 1;

    auto output = torch::empty({n, out_channels, output_depth, output_height, output_width},
                              torch::device("cuda"), torch::kFloat32);

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    int threads_per_block = 256;
    int total_elements = n * out_channels * output_depth * output_height * output_width;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    conv3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        n,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride,
        padding,
        dilation,
        output_depth,
        output_height,
        output_width,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor custom_conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
"""

# Compile the CUDA code
conv3d_module = load_inline(
    name='custom_conv3d',
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=['custom_conv3d_forward'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        kernel_depth, kernel_height, kernel_width = kernel_size
        weight_shape = (out_channels, in_channels // groups, kernel_depth, kernel_height, kernel_width)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if self.bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias_param = None

        # Initialize parameters like PyTorch's Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        if self.bias_param is None:
            bias_tensor = torch.empty(0, device=x.device)
        else:
            bias_tensor = self.bias_param
        return conv3d_module.custom_conv3d_forward(
            x, self.weight, bias_tensor,
            self.stride, self.padding, self.dilation, self.groups
        )

def get_inputs():
    batch_size = 16
    in_channels = 3
    width = 64
    height = 64
    depth = 64
    x = torch.rand(batch_size, in_channels, width, height, depth).cuda()
    return [x]

def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]

if __name__ == "__main__":
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    width = 64
    height = 64
    depth = 64

    # Original model
    model = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0).cuda()
    # New model
    model_new = ModelNew(in_channels, out_channels, kernel_size, stride=1, padding=0).cuda()

    # Copy weights and bias for comparison
    with torch.no_grad():
        model_new.weight.copy_(model.weight)
        if model_new.bias_param is not None:
            model_new.bias_param.copy_(model.bias)

    x = torch.randn(batch_size, in_channels, width, height, depth, device='cuda', requires_grad=True)
    y = model(x)
    y_new = model_new(x)
    assert torch.allclose(y, y_new, atol=1e-5), "Outputs are not close enough!"