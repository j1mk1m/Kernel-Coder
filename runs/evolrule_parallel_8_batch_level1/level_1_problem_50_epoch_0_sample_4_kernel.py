import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel code
custom_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define K 11
#define STRIDE 4
#define PADDING 2

template <typename T>
__global__ void custom_conv2d_kernel(
    const T* input,  // shape (N, C_in, H_padded, W_padded)
    const T* weight, // shape (C_out, C_in, K, K)
    const T* bias,   // shape (C_out, )
    T* output,       // shape (N, C_out, H_out, W_out)
    int N, int C_in, int H_in_padded, int W_in_padded,
    int C_out,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    int j = idx % W_out;
    int rem = idx / W_out;
    int i = rem % H_out;
    rem /= H_out;
    int c_out = rem % C_out;
    rem /= C_out;
    int n = rem;

    T sum = 0.0;
    for (int k = 0; k < C_in; ++k) {
        for (int s = 0; s < K; ++s) {
            for (int t = 0; t < K; ++t) {
                int in_row = i * STRIDE + s;
                int in_col = j * STRIDE + t;

                int input_offset = 
                    n * C_in * H_in_padded * W_in_padded 
                    + k * H_in_padded * W_in_padded 
                    + in_row * W_in_padded 
                    + in_col;

                T in_val = input[input_offset];

                int weight_offset = 
                    c_out * C_in * K * K 
                    + k * K * K 
                    + s * K 
                    + t;

                T w_val = weight[weight_offset];

                sum += in_val * w_val;
            }
        }
    }

    sum += bias[c_out];

    int output_offset = 
        n * C_out * H_out * W_out 
        + c_out * H_out * W_out 
        + i * W_out 
        + j;

    output[output_offset] = sum;
}

at::Tensor custom_conv2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);

    int H_padded = H_in + 2 * PADDING;
    int W_padded = W_in + 2 * PADDING;

    int H_out = (H_in + 2 * PADDING - K) / STRIDE + 1;
    int W_out = (W_in + 2 * PADDING - K) / STRIDE + 1;

    at::Tensor input_padded = at::constant_pad_nd(input, {PADDING, PADDING, PADDING, PADDING});

    at::Tensor output = at::empty({N, C_out, H_out, W_out}, input.options());

    int block_size = 256;
    int grid_size = (N * C_out * H_out * W_out + block_size - 1) / block_size;

    custom_conv2d_kernel<float><<<grid_size, block_size>>>(
        input_padded.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_padded, W_padded,
        C_out,
        H_out, W_out
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        abort();
    }

    return output;
}
"""

custom_conv2d_cpp = """
at::Tensor custom_conv2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias
);
"""

# Compile the CUDA code
custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=custom_conv2d_cpp,
    cuda_sources=custom_conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.bias = nn.Parameter(torch.empty(96))
        # Initialize parameters with the same method as PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return custom_conv2d.custom_conv2d_cuda(x, self.weight, self.bias)