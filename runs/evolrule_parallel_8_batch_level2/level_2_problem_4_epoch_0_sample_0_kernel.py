import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused Conv2d + Mish + Mish kernel
fused_conv_mish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define MISH(x) (x * tanh(log(1 + exp(x))))  // Mish activation function

__global__ void fused_conv_mish_forward(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> output,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> bias,
    int kernel_size, int out_channels, int out_h, int out_w) {

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;

    float acc = (bias[c] if bias.size(0) > 0 else 0.0f);

    for (int i = 0; i < C; ++i) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h + kh;
                int w_in = w + kw;
                if (h_in < H && w_in < W) {
                    acc += weight[c][i][kh][kw] * input[n][i][h_in][w_in];
                }
            }
        }
    }

    // Apply Mish activation twice
    float mish1 = acc * tanh(log(1 + exp(acc)));
    float mish2 = mish1 * tanh(log(1 + exp(mish1)));
    
    output[n][c][h][w] = mish2;
}

torch::Tensor fused_conv_mish_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int out_channels, int out_h, int out_w) {
    auto output = torch::empty({input.size(0), out_channels, out_h, out_w}, input.options());
    
    dim3 threads(kernel_size, kernel_size);
    dim3 blocks(input.size(0), out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_mish_forward", ([&] {
        fused_conv_mish_forward<<<blocks, threads>>>(
            input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            kernel_size, out_channels, out_h, out_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Load fused CUDA operator
fused_conv_mish = load_inline(
    name="fused_conv_mish",
    cuda_sources=fused_conv_mish_source,
    functions=["fused_conv_mish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.fused_conv_mish = fused_conv_mish

    def forward(self, x):
        # Calculate output dimensions
        padding = (self.kernel_size - 1) // 2  # Assuming same padding
        H_out = (x.size(2) + 2 * padding - self.kernel_size) + 1
        W_out = (x.size(3) + 2 * padding - self.kernel_size) + 1
        
        return self.fused_conv_mish.fused_conv_mish_cuda(
            x, 
            self.weight, 
            self.bias, 
            self.kernel_size, 
            self.out_channels,
            H_out,
            W_out
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]