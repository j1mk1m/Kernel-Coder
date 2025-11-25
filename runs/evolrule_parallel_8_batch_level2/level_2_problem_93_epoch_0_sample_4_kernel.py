import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

        # Define the fused element-wise CUDA kernel
        fused_elementwise_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_elementwise_kernel(
            const float* x,
            float add_val,
            float multiply_val,
            float* out,
            int size) 
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float temp1 = x[idx] + add_val;
                float temp2 = fminf(temp1, 0.0f);
                const float a = 0.7978845608f; // sqrt(2/pi)
                float arg = a * (temp2 + 0.044715f * temp2 * temp2 * temp2);
                float tanh_val = tanhf(arg);
                float gelu_val = 0.5f * temp2 * (1.0f + tanh_val);
                out[idx] = gelu_val * multiply_val;
            }
        }

        torch::Tensor fused_elementwise_cuda(torch::Tensor x, float add_val, float multiply_val) {
            auto size = x.numel();
            auto out = torch::empty_like(x);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_elementwise_kernel<<<num_blocks, block_size>>>(
                x.data_ptr<float>(),
                add_val,
                multiply_val,
                out.data_ptr<float>(),
                size
            );

            return out;
        }
        """

        fused_elementwise_cpp_source = (
            "torch::Tensor fused_elementwise_cuda(torch::Tensor x, float add_val, float multiply_val);"
        )

        # Compile the CUDA kernel
        self.fused_elementwise = load_inline(
            name="fused_elementwise",
            cpp_sources=fused_elementwise_cpp_source,
            cuda_sources=fused_elementwise_source,
            functions=["fused_elementwise_cuda"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"]
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_elementwise.fused_elementwise_cuda(x, self.add_value, self.multiply_value)