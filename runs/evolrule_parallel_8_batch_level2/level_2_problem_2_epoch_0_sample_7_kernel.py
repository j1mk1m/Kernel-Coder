import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

        # Define CUDA kernel for element-wise bias addition
        add_bias_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template<typename T>
        __global__ void add_bias_kernel(const T* input, const T* bias, T* output, int batch, int channels, int height, int width) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch * channels * height * width) {
                int c = (idx / (height * width)) % channels;  // Compute channel index
                output[idx] = input[idx] + bias[c];
            }
        }

        at::Tensor add_bias_cuda(at::Tensor input, at::Tensor bias) {
            const int batch = input.size(0);
            const int channels = input.size(1);
            const int height = input.size(2);
            const int width = input.size(3);
            const int size = batch * channels * height * width;

            const int threads_per_block = 256;
            const int blocks = (size + threads_per_block - 1) / threads_per_block;

            at::Tensor output = at::empty_like(input);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "add_bias_cuda", ([&] {
                add_bias_kernel<scalar_t><<<blocks, threads_per_block>>>(
                    input.data<scalar_t>(),
                    bias.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch, channels, height, width
                );
            }));

            cudaDeviceSynchronize();
            return output;
        }
        """

        add_bias_cpp = (
            "at::Tensor add_bias_cuda(at::Tensor input, at::Tensor bias);"
        )

        # Compile the bias addition kernel
        self.add_bias = load_inline(
            name="add_bias",
            cpp_sources=add_bias_cpp,
            cuda_sources=add_bias_source,
            functions=["add_bias_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.add_bias.add_bias_cuda(x, self.bias)  # Custom kernel for bias addition
        x = torch.clamp(x, min=0.0, max=1.0)          # Must remain
        x = x * self.scaling_factor                    # Must remain
        x = torch.clamp(x, min=0.0, max=1.0)          # Must remain
        x = x / self.scaling_factor                    # Must remain
        return x

# The original code's get_inputs and get_init_inputs remain unchanged