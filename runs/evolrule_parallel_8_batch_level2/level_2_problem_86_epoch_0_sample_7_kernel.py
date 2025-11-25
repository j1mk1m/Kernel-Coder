import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Linear (matmul + bias) + division + GELU
fused_linear_div_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256

template <typename scalar_t>
__global__ void fused_linear_div_gelu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int input_size,
    const int output_size,
    const scalar_t divisor) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * output_size) return;

    const int row = idx / output_size;
    const int col = idx % output_size;

    scalar_t sum = bias[col];
    for (int k = 0; k < input_size; ++k) {
        sum += input[row * input_size + k] * weight[col * input_size + k];
    }
    sum /= divisor;

    // GELU approximation: 0.5 * x * (1 + tanh[sqrt(2/pi)*(x + 0.044715x^3)])
    scalar_t x = sum;
    scalar_t inner = sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    scalar_t tanh_inner = tanh(inner);
    output[idx] = 0.5f * x * (1.0f + tanh_inner);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_linear_div_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    const float divisor) {

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = weight.size(0); // output_size is first dim of weight

    auto output = torch::empty({batch_size, output_size}, input.options());

    const int total_elements = batch_size * output_size;
    const dim3 blocks((total_elements + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_linear_div_gelu_cuda", ([&] {
        fused_linear_div_gelu_kernel<scalar_t><<<blocks, THREADS>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            input_size,
            output_size,
            divisor);
    }));

    return std::make_tuple(output, weight, bias);
}
"""

# Compile the fused CUDA kernel
fused_linear_div_gelu = load_inline(
    name="fused_linear_div_gelu",
    cpp_sources="",
    cuda_sources=fused_linear_div_gelu_source,
    functions=["fused_linear_div_gelu_cuda"],
    verbose=True,
    extra_cflags=["-DDEBUG"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.divisor = divisor

        # Initialize weights and bias like original Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = fused_linear_div_gelu

    def forward(self, x):
        # The fused kernel returns output, weight, bias - we only need output
        output, _, _ = self.fused_op.fused_linear_div_gelu_cuda(
            x, self.weight, self.bias, self.divisor
        )
        return output

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, output_size, divisor]

# Constants from original code
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0