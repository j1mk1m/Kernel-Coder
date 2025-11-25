import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for im2col transformation
im2col_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void im2col_kernel(
    const float* input,
    float* output,
    int N, int C_in, int L_padded, int K,
    int stride, int dilation, int L_out,
    int output_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    int row = idx / (C_in * K);
    int col = idx % (C_in * K);

    int n = row / L_out;
    int l_out = row % L_out;

    int c_in = col / K;
    int k = col % K;

    int input_pos = l_out * stride + k * dilation;

    if (input_pos < 0 || input_pos >= L_padded) {
        output[idx] = 0.0f;
        return;
    }

    int input_offset = n * C_in * L_padded + c_in * L_padded + input_pos;
    output[idx] = input[input_offset];
}

torch::Tensor im2col_cuda(torch::Tensor input, int kernel_size, int stride, int dilation) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto L_padded = input.size(2);

    int kernel_size_eff = kernel_size + (kernel_size - 1) * (dilation - 1);
    int L_out = (L_padded - kernel_size_eff) / stride + 1;

    int im2col_channels = C_in * kernel_size;
    int im2col_size = N * L_out;
    int output_size = im2col_size * im2col_channels;

    auto im2col = torch::empty({im2col_size, im2col_channels}, input.options());

    dim3 block(256);
    dim3 grid((output_size + block.x - 1) / block.x);

    im2col_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        im2col.data_ptr<float>(),
        N, C_in, L_padded, kernel_size,
        stride, dilation, L_out, output_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return im2col;
}
"""

# Compile the inline CUDA code
im2col_cuda = load_inline(
    name="im2col_cuda",
    cpp_sources="",
    cuda_sources=im2col_source,
    functions=["im2col_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == 1, "Only groups=1 is supported"
        assert dilation == 1, "Only dilation=1 is supported"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Weight initialization (matches nn.Conv1d)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Pad input tensor
        padded_x = F.pad(x, (self.padding, self.padding), 'constant', 0)
        
        # Compute im2col representation
        im2col = im2col_cuda(padded_x, self.kernel_size, self.stride, self.dilation)
        
        # Reshape and compute matrix multiplication
        N = x.size(0)
        weight_reshaped = self.weight.view(self.out_channels, -1)
        output = torch.mm(im2col, weight_reshaped.t())
        
        # Reshape to output dimensions
        L_out = (padded_x.size(2) - self.kernel_size) // self.stride + 1
        output = output.view(N, L_out, self.out_channels).transpose(1, 2)
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
            
        return output