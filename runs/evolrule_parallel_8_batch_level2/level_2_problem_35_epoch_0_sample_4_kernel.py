import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        
        # Fused subtract + hardswish kernel
        fused_subtract_hardswish_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_subtract_hardswish_kernel(const float* input, const float subtract_val, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float z = input[idx] - subtract_val;
                float temp = z + 3.0f;
                temp = temp < 0.0f ? 0.0f : temp;
                temp = temp > 6.0f ? 6.0f : temp;
                output[idx] = z * temp / 6.0f;
            }
        }

        torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_val) {
            auto size = input.numel();
            auto output = torch::empty_like(input);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_subtract_hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), subtract_val, output.data_ptr<float>(), size);
            return output;
        }
        """
        fused_subtract_hardswish_cpp = """
        torch::Tensor fused_subtract_hardswish_cuda(torch::Tensor input, float subtract_val);
        """
        
        # Mish kernel
        mish_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void mish_kernel(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = input[idx];
                float exp_x = expf(x);
                float softplus_x = logf(1.0f + exp_x);
                float tanh_softplus = tanhf(softplus_x);
                output[idx] = x * tanh_softplus;
            }
        }

        torch::Tensor mish_cuda(torch::Tensor input) {
            auto size = input.numel();
            auto output = torch::empty_like(input);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
            return output;
        }
        """
        mish_cpp = """
        torch::Tensor mish_cuda(torch::Tensor input);
        """
        
        # Compile the fused kernel
        self.fused_subtract_hardswish = load_inline(
            name="fused_subtract_hardswish",
            cpp_sources=fused_subtract_hardswish_cpp,
            cuda_sources=fused_subtract_hardswish_source,
            functions=["fused_subtract_hardswish_cuda"],
            verbose=True
        )
        
        # Compile the mish kernel
        self.mish_cuda = load_inline(
            name="mish_cuda",
            cpp_sources=mish_cpp,
            cuda_sources=mish_source,
            functions=["mish_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_subtract_hardswish.fused_subtract_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        x = self.mish_cuda.mish_cuda(x)
        return x

# The original functions remain unchanged
def get_inputs():
    batch_size = 128
    in_channels = 64
    height = width = 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    subtract_value = 0.5
    pool_kernel_size = 2
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]