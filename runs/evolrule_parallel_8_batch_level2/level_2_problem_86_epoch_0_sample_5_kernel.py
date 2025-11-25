import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for (Linear + division + GELU)
fused_linear_div_gelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// GELU approximation function
__forceinline__ __device__ float fast_gelu(float x) {
    return x * (0.5f * (1.0f + __nv_fast_tanh(M_1_SQRT2 * (x + 0.044715f * x * x * x))));
}

// Fused kernel: GEMM + division + GELU
__global__ void fused_linear_div_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch_size,
    int input_size,
    int output_size,
    float divisor
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * output_size) return;

    int row = tid / output_size;
    int col = tid % output_size;

    float sum = bias[col];
    for (int k = 0; k < input_size; ++k) {
        sum += input[row * input_size + k] * weight[col * input_size + k];
    }

    sum /= divisor;
    output[tid] = fast_gelu(sum);
}

torch::Tensor fused_linear_div_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int batch_size,
    int output_size
) {
    const int num_elements = batch_size * output_size;
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    auto output = torch::empty({batch_size, output_size}, input.options());

    fused_linear_div_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input.size(1),
        output_size,
        divisor
    );

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor fused_linear_div_gelu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    int batch_size,
    int output_size
);
"""

# Load the fused kernel
fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=cpp_source,
    cuda_sources=fused_linear_div_gelu_source,
    functions=["fused_linear_div_gelu"],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_75']
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_kernel = fused_kernel

    def forward(self, x):
        # Extract parameters for fused kernel
        weight = self.linear.weight.t().contiguous()  # Transposed for correct GEMM
        bias = self.linear.bias
        batch_size = x.size(0)
        output_size = self.linear.out_features

        return self.fused_kernel.fused_linear_div_gelu(
            x, weight, bias, self.divisor, batch_size, output_size
        )

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, output_size, divisor]