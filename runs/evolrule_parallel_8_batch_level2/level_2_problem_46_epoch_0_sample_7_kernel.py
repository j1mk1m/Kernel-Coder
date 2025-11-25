import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for Conv2D + Subtract + Tanh + Subtract + AvgPool
fused_conv_sub_tanh_sub_pool_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_conv_sub_tanh_sub_pool(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> bias,
    scalar_t sub1, scalar_t sub2,
    int kernel_size, int pool_kernel,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output) {

    const int B = input.size(0);
    const int Cin = input.size(1);
    const int Hin = input.size(2);
    const int Win = input.size(3);
    const int Cout = weight.size(0);
    const int Hout = output.size(2);
    const int Wout = output.size(3);

    const int padding = (kernel_size - 1) / 2;
    const int stride_conv = 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * Cout * Hout * Wout) return;

    int w_out = idx % Wout;
    int h_out = (idx / Wout) % Hout;
    int c_out = (idx / (Wout * Hout)) % Cout;
    int b = idx / (Wout * Hout * Cout);

    int h_start = h_out * pool_kernel;
    int w_start = w_out * pool_kernel;

    scalar_t sum = 0.0;
    for (int ph = 0; ph < pool_kernel; ph++) {
        for (int pw = 0; pw < pool_kernel; pw++) {
            int h_conv = h_start + ph;
            int w_conv = w_start + pw;

            scalar_t conv_val = bias[c_out];
            for (int i = 0; i < Cin; i++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int h_in = h_conv - padding + kh;
                        int w_in = w_conv - padding + kw;
                        if (h_in >= 0 && h_in < Hin && w_in >= 0 && w_in < Win) {
                            conv_val += input[b][i][h_in][w_in] * weight[c_out][i][kh][kw];
                        }
                    }
                }
            }
            conv_val = tanh(conv_val - sub1);
            sum += conv_val;
        }
    }

    scalar_t avg_val = sum / (pool_kernel * pool_kernel);
    output[b][c_out][h_out][w_out] = avg_val - sub2;
}

torch::Tensor fused_conv_sub_tanh_sub_pool_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float sub1, float sub2,
    int kernel_size,
    int pool_kernel
) {
    const auto B = input.size(0);
    const auto Cout = weight.size(0);
    const auto Hin = input.size(2);
    const auto Win = input.size(3);

    const int padding = (kernel_size - 1) / 2;
    const int stride_conv = 1;

    int H_conv = (Hin + 2*padding - kernel_size)/stride_conv + 1;
    int W_conv = (Win + 2*padding - kernel_size)/stride_conv + 1;

    int H_out = H_conv / pool_kernel;
    int W_out = W_conv / pool_kernel;

    auto output = torch::empty({B, Cout, H_out, W_out}, input.options());

    const int total_elements = B * Cout * H_out * W_out;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_sub_tanh_sub_pool_cuda", ([&] {
        fused_conv_sub_tanh_sub_pool<scalar_t><<<blocks, threads_per_block>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
            sub1, sub2,
            kernel_size, pool_kernel,
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

# Compile the fused CUDA kernel
fused_conv_sub_tanh_sub_pool = load_inline(
    name="fused_conv_sub_tanh_sub_pool",
    cuda_sources=fused_conv_sub_tanh_sub_pool_source,
    functions=["fused_conv_sub_tanh_sub_pool_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        kernel_size = self.conv.kernel_size[0]
        
        return fused_conv_sub_tanh_sub_pool_cuda(
            x, weight, bias,
            self.subtract1_value, self.subtract2_value,
            kernel_size, self.kernel_size_pool
        )