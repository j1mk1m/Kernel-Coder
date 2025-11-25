import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel for linear (matmul + bias) + ReLU + division by constant
fused_linear_relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

template <typename scalar_t>
__global__ void fused_linear_relu_div_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    scalar_t divisor) {

    const int batch_id = blockIdx.x;
    const int out_id = threadIdx.x;

    scalar_t sum = 0.0;
    for (int in_id = 0; in_id < in_features; ++in_id) {
        sum += input[batch_id * in_features + in_id] * weight[in_id * out_features + out_id];
    }

    sum += bias[out_id];
    sum = fmaxf(sum, 0.0);  // Apply ReLU
    output[batch_id * out_features + out_id] = sum / divisor;
}

torch::Tensor fused_linear_relu_div_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float divisor) {

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(1);

    dim3 blocks(batch_size);
    dim3 threads(out_features);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_linear_relu_div_forward", ([&] {
        fused_linear_relu_div_forward<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            divisor);
    }));

    return output;
}
"""

fused_linear_relu_div_cpp_source = (
    "torch::Tensor fused_linear_relu_div_forward_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, float divisor);"
)

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_linear_relu_div",
    cpp_sources=fused_linear_relu_div_cpp_source,
    cuda_sources=fused_linear_relu_div_source,
    functions=["fused_linear_relu_div_forward_cuda"],
    verbose=True,
    extra_cflags=["-gencode=arch=compute_80,code=sm_80"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.divisor = divisor
        self.fused_op = fused_ops

        # Initialize weights and bias like original nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.linalg.matrix_norm(self.weight, ord=2, dim=(0,1)).item()
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = torch.empty_like(x[:, :self.bias.size(0)])
        return self.fused_op.fused_linear_relu_div_forward_cuda(
            x, self.weight.t(), self.bias, output, self.divisor
        )

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, divisor]