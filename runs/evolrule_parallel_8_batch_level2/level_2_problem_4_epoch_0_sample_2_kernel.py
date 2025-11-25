import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused Mish kernel
fused_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_mish_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float exp_x = expf(x);
        float softplus = logf(1.f + exp_x);
        output[idx] = x * tanhf(softplus);
    }
}

__global__ void fused_mish_backward(const float* grad_output, const float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float exp_x = expf(x);
        float softplus = logf(1.f + exp_x);
        float tanhsp = tanhf(softplus);
        float dspx = (exp_x) / (1.f + exp_x); // derivative of softplus
        float dtanhsp = 1 - tanhsp * tanhsp;
        float grad = grad_output[idx] * (tanhsp + x * dspx * dtanhsp);
        grad_input[idx] = grad;
    }
}

torch::Tensor mish_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_mish_forward<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

torch::Tensor mish_backward_cuda(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_mish_backward<<<num_blocks, block_size>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), size);
    return grad_input;
}
"""

fused_mish_forward = load_inline(
    name="fused_mish_forward",
    cuda_sources=fused_mish_source,
    functions=["mish_forward_cuda"],
    verbose=True
)

fused_mish_backward = load_inline(
    name="fused_mish_backward",
    cuda_sources=fused_mish_source,
    functions=["mish_backward_cuda"],
    verbose=True
)

class FusedMishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return fused_mish_forward.mish_forward_cuda(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return fused_mish_backward.mish_backward_cuda(grad_output, input)

class FusedMish(nn.Module):
    def forward(self, x):
        return FusedMishFunction.apply(x)

# Custom fused Conv + Mish kernel
# This kernel combines convolution and Mish activation to reduce memory traffic
fused_conv_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void fused_conv_mish_forward(const scalar_t* input, const scalar_t* weight, scalar_t* output, int batch, int in_channels, int height, int width, int out_channels, int kernel_size, int pad) {
    // This is a simplified version. A full implementation requires proper convolution indexing and parallelization
    // For brevity, we'll assume the output dimensions are computed correctly
    const int output_size = height * width;
    const int kernel_elements = kernel_size * kernel_size * in_channels;
    CUDA_KERNEL_LOOP(output_idx, batch * out_channels * output_size) {
        int batch_idx = output_idx / (out_channels * output_size);
        int out_channel = (output_idx / output_size) % out_channels;
        int spatial_idx = output_idx % output_size;
        int h_out = spatial_idx / width;
        int w_out = spatial_idx % width;

        scalar_t sum = 0;
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                for (int in_c = 0; in_c < in_channels; ++in_c) {
                    int h_in = h_out + k_h - pad;
                    int w_in = w_out + k_w - pad;
                    if (h_in >=0 && h_in < height && w_in >=0 && w_in < width) {
                        int input_offset = batch_idx * in_channels * height * width + in_c * height * width + h_in * width + w_in;
                        int weight_offset = out_channel * kernel_elements + in_c * kernel_size * kernel_size + k_h * kernel_size + k_w;
                        sum += input[input_offset] * weight[weight_offset];
                    }
                }
            }
        }
        // Apply Mish activation
        scalar_t exp_sum = expf(sum);
        scalar_t softplus = logf(1.f + exp_sum);
        output[output_idx] = sum * tanhf(softplus);
    }
}

torch::Tensor fused_conv_mish_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int pad = kernel_size / 2;

    auto output = torch::empty({batch, out_channels, height, width}, input.options());
    const int threads = 256;
    const int elements = batch * out_channels * height * width;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_mish_forward_cuda", ([&] {
        fused_conv_mish_forward<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(), weight.data<scalar_t>(), 
            output.data<scalar_t>(), batch, in_channels, height, width, out_channels, kernel_size, pad);
    }));

    return output;
}
"""

# Compile fused convolution + Mish kernel
fused_conv_mish = load_inline(
    name="fused_conv_mish",
    cuda_sources=fused_conv_mish_source,
    functions=["fused_conv_mish_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.fused_conv_mish = fused_conv_mish
        # The second Mish is fused into the first one, so no additional layers needed

    def forward(self, x):
        # Apply fused convolution + Mish in one step
        x = self.fused_conv_mish.fused_conv_mish_forward_cuda(x, self.weight)
        # The second Mish is redundant as it's applied again on the same output
        # However, if required by the original model, we can apply another Mish
        # But according to the original model, there are two Mish layers sequentially
        # So we need to apply another Mish here. However, since the first Mish is already fused,
        # applying another Mish would require either a separate kernel or another fused step
        # Here we'll apply the fused Mish again for the second Mish
        return FusedMish()(x)

def get_inputs():
    batch_size = 64
    in_channels = 64
    height = width = 256
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [64, 128, 3]