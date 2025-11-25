import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch import fx
import torch._dynamo as torchdynamo

class FusedGemmMulLeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, multiplier, negative_slope):
        ctx.save_for_backward(input, weight, bias)
        ctx.multiplier = multiplier
        ctx.negative_slope = negative_slope

        # Compute Gemm (input * weight + bias)
        gemm_out = torch.addmm(bias, input, weight.t())
        # Apply scaling
        scaled = gemm_out * multiplier
        # Apply LeakyReLU
        output = torch.where(scaled > 0, scaled, scaled * negative_slope)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        multiplier = ctx.multiplier
        negative_slope = ctx.negative_slope

        # Compute d_leaky_relu
        grad_scaled = grad_output.clone()
        grad_scaled[grad_output < 0] *= negative_slope

        # Apply gradient scaling from multiplier
        grad_scaled *= multiplier

        # Compute gradients
        grad_input = grad_scaled.mm(weight)
        grad_weight = grad_scaled.t().mm(input)
        grad_bias = grad_scaled.sum(0)
        return grad_input, grad_weight, grad_bias, None, None

class FusedGemmMulLeakyReLU(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=self.negative_slope, nonlinearity="leaky_relu")
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / torch.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return FusedGemmMulLeakyReLUFunction.apply(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.fused_layer = FusedGemmMulLeakyReLU(in_features, out_features, multiplier, negative_slope)

    @torchdynamo.optimize("inductor")
    def forward(self, x):
        return self.fused_layer(x)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1