import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused activation CUDA kernels
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float z = x + 3.0f;
        float relu6_z = fmaxf(0.0f, fminf(z, 6.0f));
        float y = x * relu6_z / 6.0f;
        output[idx] = fmaxf(0.0f, y);
    }
}

__global__ void fused_backward(const float* grad_output, const float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float go = grad_output[idx];
        float z = x + 3.0f;
        float relu6_z = fmaxf(0.0f, fminf(z, 6.0f));
        float y = x * relu6_z / 6.0f;
        float d_relu6_dz = (z > 0.0f && z < 6.0f) ? 1.0f : 0.0f;
        float dy_dx = (relu6_z + x * d_relu6_dz) / 6.0f;
        float d_relu_y = (y > 0.0f) ? 1.0f : 0.0f;
        grad_input[idx] = go * d_relu_y * dy_dx;
    }
}

torch::Tensor fused_forward_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_forward<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}

std::tuple<torch::Tensor> fused_backward_cuda(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::zeros_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_backward<<<num_blocks, block_size>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), size);
    return std::make_tuple(grad_input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward_cuda", &fused_forward_cuda, "Fused activation forward");
    m.def("fused_backward_cuda", &fused_backward_cuda, "Fused activation backward");
}
"""

# Compile the fused activation CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cuda_sources=fused_activation_source,
    functions=["fused_forward_cuda", "fused_backward_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "--expt-extended-lambda"],
    extra_ldflags=[""],
)

class FusedActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return fused_activation.fused_forward_cuda(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input_tuple = fused_activation.fused_backward_cuda(grad_output, input)
        return grad_input_tuple[0]  # Unwrap the tuple

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = FusedActivation.apply(x)
        return x