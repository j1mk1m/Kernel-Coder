import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

        # Define the fused CUDA kernel for subtraction and Mish activation
        fused_subtract_mish_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_subtract_mish_kernel(
            const float* input,
            float* output,
            float sub_val1,
            float sub_val2,
            int size) 
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            float temp = input[idx] - (sub_val1 + sub_val2);

            // Compute softplus(temp)
            float softplus_val;
            if (temp > 20.0f) {
                softplus_val = temp;
            } else if (temp < -20.0f) {
                softplus_val = 0.0f;
            } else {
                softplus_val = logf(1.0f + expf(temp));
            }

            // Compute tanh(softplus_val)
            float tanh_val = tanhf(softplus_val);

            // Compute mish = temp * tanh_val
            output[idx] = temp * tanh_val;
        }

        torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float sub_val1, float sub_val2) {
            auto output = torch::empty_like(input);
            const int size = input.numel();

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_subtract_mish_kernel<<<num_blocks, block_size>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                sub_val1,
                sub_val2,
                size
            );

            return output;
        }
        """

        fused_subtract_mish_cpp_source = """
        torch::Tensor fused_subtract_mish_cuda(torch::Tensor input, float sub_val1, float sub_val2);
        """

        # Compile the fused CUDA kernel
        self.fused_subtract_mish = load_inline(
            name="fused_subtract_mish",
            cpp_sources=fused_subtract_mish_cpp_source,
            cuda_sources=fused_subtract_mish_source,
            functions=["fused_subtract_mish_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_subtract_mish.fused_subtract_mish_cuda(
            x, self.subtract_value_1, self.subtract_value_2
        )
        return x

# Initialization and input functions remain unchanged
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]