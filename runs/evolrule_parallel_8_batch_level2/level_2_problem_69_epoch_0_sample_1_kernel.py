import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Define fused activation CUDA kernel
        fused_activation_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_activation_kernel(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                float clamped = fmaxf(x + 3.0f, 0.0f);
                clamped = fminf(clamped, 6.0f);
                float y = x * clamped / 6.0f;
                output[idx] = fmaxf(y, 0.0f);
            }
        }

        torch::Tensor fused_activation_cuda(torch::Tensor input) {
            auto output = torch::empty_like(input);
            int size = input.numel();
            
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_activation_kernel<<<num_blocks, block_size>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                size
            );

            return output;
        }
        """

        fused_activation_cpp_source = (
            "torch::Tensor fused_activation_cuda(torch::Tensor input);"
        )

        # Compile the fused activation kernel
        self.fused_activation = load_inline(
            name="fused_activation",
            cpp_sources=fused_activation_cpp_source,
            cuda_sources=fused_activation_source,
            functions=["fused_activation_cuda"],
            verbose=False,
            extra_cflags=[""],
            extra_ldflags=[""],
        )

    def forward(self, x):
        x = self.conv(x)
        # Apply fused activation instead of separate hardswish and ReLU
        return self.fused_activation.fused_activation_cuda(x)

# Original get_inputs and get_init_inputs remain unchanged
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]