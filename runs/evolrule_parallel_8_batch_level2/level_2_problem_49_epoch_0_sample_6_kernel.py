import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation CUDA kernel (softmax followed by sigmoid)
fused_activation_source = """
#include <torch/extension.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void fused_softmax_sigmoid_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int D,
    int H,
    int W
) {
    int block_idx = blockIdx.x;
    int b = block_idx / (D * H * W);
    int rem = block_idx % (D * H * W);
    int d = rem / (H * W);
    rem %= (H * W);
    int h = rem / W;
    int w = rem % W;

    extern __shared__ float sdata[];
    float* exp_vals = sdata;

    int c = threadIdx.x;

    float exp_val = 0.0f;
    if (c < channels) {
        int offset = b * channels * D * H * W
                    + c * D * H * W
                    + d * H * W
                    + h * W
                    + w;
        float x = input[offset];
        exp_val = expf(x);
    }

    exp_vals[threadIdx.x] = (c < channels) ? exp_val : 0.0f;
    __syncthreads();

    // Reduction step
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            exp_vals[threadIdx.x] += exp_vals[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum_exp = exp_vals[0];
    if (threadIdx.x == 0) {
        if (sum_exp == 0.0f)
            sum_exp = 1e-8f; // Prevent division by zero
    }
    __syncthreads();

    if (c < channels) {
        float softmax_val = exp_val / sum_exp;
        float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
        output[offset] = sigmoid_val;
    }
}

torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    dim3 block(channels);
    dim3 grid(batch_size * D * H * W);

    int shared_mem_size = channels * sizeof(float);

    fused_softmax_sigmoid_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, D, H, W
    );

    return output;
}
"""

fused_activation_cpp_source = """
torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input);
"""

# Compile the fused activation kernel
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_softmax_sigmoid_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            bias=bias
        )
        self.fused_activation = fused_activation  # Store the compiled kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply fused softmax and sigmoid
        x = self.fused_activation.fused_softmax_sigmoid_cuda(x)
        return x

def get_inputs():
    batch_size = 16
    in_channels = 32
    D, H, W = 16, 32, 32
    return [torch.rand(batch_size, in_channels, D, H, W, device='cuda')]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 1]