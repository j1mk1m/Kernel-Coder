import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class FusedLinearSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # Custom fused kernel for Linear followed by Sigmoid
        output = fused_linear_sigmoid(input, weight, bias)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        # Compute gradients using another fused kernel
        grad_input, grad_weight, grad_bias = fused_linear_sigmoid_backward(
            grad_output, input, weight, bias, output
        )
        return grad_input, grad_weight, grad_bias

def fused_linear_sigmoid(input, weight, bias):
    # Inline CUDA kernel for fused linear + sigmoid
    output = torch.empty(
        (input.size(0), weight.size(0)),
        device=input.device,
        dtype=input.dtype
    )
    # Launch kernel
    n = input.numel()
    threads = 256
    blocks = (n + threads - 1) // threads
    fused_linear_sigmoid_kernel[blocks, threads](
        input.contiguous(), weight.t().contiguous(), bias.contiguous(),
        output
    )
    return output

def fused_linear_sigmoid_backward(grad_output, input, weight, bias, output):
    # Compute gradients in a fused manner
    grad_input = torch.empty_like(input)
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty_like(bias)
    n = grad_output.numel()
    threads = 256
    blocks = (n + threads - 1) // threads
    fused_backward_kernel[blocks, threads](
        grad_output.contiguous(), input.contiguous(), weight.contiguous(),
        output.contiguous(), grad_input, grad_weight, grad_bias
    )
    return grad_input, grad_weight.t(), grad_bias

# Define CUDA kernels
fused_linear_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_sigmoid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int batch_size, int in_features, int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int sample = idx / out_features;
    int out_feat = idx % out_features;

    float sum = bias[out_feat];
    for (int i = 0; i < in_features; ++i) {
        sum += input[sample * in_features + i] * weight[out_feat * in_features + i];
    }
    output[idx] = 1.0f / (1.0f + expf(-sum));
}

__global__ void fused_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* weight,
    const float* output,
    float* grad_input,
    float* grad_weight,
    float* grad_bias,
    int batch_size, int in_features, int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int sample = idx / out_features;
    int out_feat = idx % out_features;

    float g = grad_output[idx] * output[idx] * (1 - output[idx]);

    // Accumulate grad_weight
    atomicAdd(&grad_weight[out_feat * in_features + i], input[sample * in_features + i] * g);

    // Accumulate grad_bias
    atomicAdd(&grad_bias[out_feat], g);

    // Compute grad_input
    for (int i = 0; i < in_features; ++i) {
        atomicAdd(&grad_input[sample * in_features + i], weight[out_feat * in_features + i] * g);
    }
}
"""

# Compile the kernels
load_inline(
    name="fused_ops",
    cuda_sources=fused_linear_sigmoid_source,
    functions=[
        "fused_linear_sigmoid_kernel",
        "fused_backward_kernel"
    ],
    extra_cuda_cflags=["-arch=sm_86"],
    verbose=True
)

class FusedLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        max_val = input.max(dim=1, keepdim=True).values
        exps = torch.exp(input - max_val)
        sum_exp = exps.sum(dim=1, keepdim=True)
        result = (max_val.squeeze() + torch.log(sum_exp.squeeze())).unsqueeze(1)
        ctx.save_for_backward(exps, sum_exp, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        exps, sum_exp, result = ctx.saved_tensors
        grad_input = exps / sum_exp * grad_output.view(-1, 1)
        return grad_input

class FusedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = FusedLinear(input_size, hidden_size, bias=True)
        self.linear2 = FusedLinear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = FusedLinearSigmoid.apply(x, self.linear1.weight, self.linear1.bias)
        x = self.linear2(x)
        x = FusedLogSumExp.apply(x)
        return x.view(-1)