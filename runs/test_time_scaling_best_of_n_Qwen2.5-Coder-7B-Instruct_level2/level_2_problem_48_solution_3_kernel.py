import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Load the custom CUDA kernels
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_forward_kernel(
    const float* input, 
    const float* weight, 
    float* output, 
    int N, int C_in, int D, int H, int W, 
    int C_out, int K_d, int K_h, int K_w
) {
    int n = blockIdx.z; // batch index
    int c_out = blockIdx.y; // output channel index
    int d = blockIdx.x / (H * W); // depth index
    int h = (blockIdx.x % (H * W)) / W; // height index
    int w = blockIdx.x % W; // width index

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k_d = 0; k_d < K_d; ++k_d) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int i_d = d + k_d - K_d / 2;
                    int i_h = h + k_h - K_h / 2;
                    int i_w = w + k_w - K_w / 2;
                    if (i_d >= 0 && i_d < D && i_h >= 0 && i_h < H && i_w >= 0 && i_w < W) {
                        int input_idx = n * C_in * D * H * W + c_in * D * H * W + i_d * H * W + i_h * W + i_w;
                        int weight_idx = c_out * C_in * K_d * K_h * K_w + c_in * K_d * K_h * K_w + k_d * K_h * K_w + k_h * K_w + k_w;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int output_idx = n * C_out * D * H * W + c_out * D * H * W + d * H * W + h * W + w;
    output[output_idx] = sum;
}

torch::Tensor conv3d_forward_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    int K_d, int K_h, int K_w
) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto C_out = weight.size(0);

    auto output = torch::zeros({N, C_out, D, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (D * H * W + block_size - 1) / block_size;

    dim3 grid(C_out, N, 1);
    dim3 block(block_size, 1, 1);

    conv3d_forward_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C_in, D, H, W, C_out, K_d, K_h, K_w);

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, int K_d, int K_h, int K_w);"
)

conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

elementwise_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_ops_kernel(
    const float* input, 
    const float* scale, 
    const float* bias, 
    float* output, 
    int N, int C, int D, int H, int W
) {
    int n = blockIdx.z; // batch index
    int c = blockIdx.y; // channel index
    int d = blockIdx.x / (H * W); // depth index
    int h = (blockIdx.x % (H * W)) / W; // height index
    int w = blockIdx.x % W; // width index

    int idx = n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
    float value = input[idx];

    // Element-wise operations
    value *= scale[c];
    value += bias[c];
    value = tanh(value);
    value = sigmoid(value);

    output[idx] = value;
}

torch::Tensor elementwise_ops_cuda(
    torch::Tensor input, 
    torch::Tensor scale, 
    torch::Tensor bias
) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (D * H * W + block_size - 1) / block_size;

    dim3 grid(C, N, 1);
    dim3 block(block_size, 1, 1);

    elementwise_ops_kernel<<<grid, block>>>(input.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), N, C, D, H, W);

    return output;
}
"""

elementwise_ops_cpp_source = (
    "torch::Tensor elementwise_ops_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);"
)

elementwise_ops = load_inline(
    name="elementwise_ops",
    cpp_sources=elementwise_ops_cpp_source,
    cuda_sources=elementwise_ops_source,
    functions=["elementwise_ops_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the new model architecture
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = conv3d
        self.scale = nn.Parameter(torch.randn(out_channels))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv.conv3d_forward_cuda(x, self.weight, kernel_size[0], kernel_size[1], kernel_size[2])
        x = x * self.scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = torch.tanh(x)
        x = x * self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 3
    out_channels = 16
    depth, height, width = 16, 64, 64
    kernel_size = (3, 3, 3)
    scaling_factor = 2
    bias_shape = (out_channels, 1, 1, 1)

    model = ModelNew(in_channels, out_channels, kernel_size, scaling_factor, bias_shape)
    inputs = get_inputs()
    outputs = model(inputs[0].cuda())
    print(outputs.shape)