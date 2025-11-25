import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
torch::Tensor fused_linear_relu_div_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void fused_linear_relu_div_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float divisor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int row = idx / out_features;
    int col = idx % out_features;

    scalar_t sum = bias[col];
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum = fmaxf(sum, 0.0f) / divisor;
    output[idx] = sum;
}

torch::Tensor fused_linear_relu_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    int num_elements = batch_size * out_features;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_linear_relu_div_cuda", ([&] {
        fused_linear_relu_div_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            divisor
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

fused_op = load_inline(
    name="fused_linear_relu_div",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_relu_div_cuda"],
    verbose=True
)

class FusedLinearReLU_DIV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, divisor):
        ctx.save_for_backward(input, weight, bias)
        ctx.divisor = divisor
        return fused_op.fused_linear_relu_div_cuda(input, weight, bias, divisor)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        divisor = ctx.divisor
        batch_size, in_features = input.shape
        _, out_features = weight.shape

        # Compute gradients using PyTorch operations for simplicity
        output = fused_op.fused_linear_relu_div_cuda(input, weight, bias, divisor)
        mask = (output > 0).float() / divisor

        grad_input = grad_output * mask.mm(weight.t())
        grad_weight = grad_output.t().mm(input * mask)
        grad_bias = (grad_output * mask).sum(0)
        
        return grad_input, grad_weight, grad_bias, None

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.divisor = divisor
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return FusedLinearReLU_DIV.apply(x, self.weight, self.bias, self.divisor)