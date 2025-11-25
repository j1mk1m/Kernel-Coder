import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

        # Define the fused CUDA kernel
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_kernel(const float* input, const float* bias, float scaling_factor, float* output, int batch, int channels, int height, int width) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch * channels * height * width) return;

            int w = idx % width;
            int h = (idx / width) % height;
            int c = (idx / (width * height)) % channels;
            int b = idx / (width * height * channels);

            float val = input[idx] + bias[c]; // Bias is (channels, 1, 1)
            val = fmaxf(0.0f, fminf(1.0f, val));
            val *= scaling_factor;
            val = fmaxf(0.0f, fminf(1.0f, val));
            val /= scaling_factor;
            output[idx] = val;
        }

        torch::Tensor fused_operation(torch::Tensor input, torch::Tensor bias, float scaling_factor) {
            input = input.contiguous();
            bias = bias.contiguous();

            // Check device
            TORCH_CHECK(input.device() == bias.device(), "Input and bias must be on the same device");

            // Check bias shape
            int input_channels = input.size(1);
            int bias_channels = bias.size(0);
            int bias_height = bias.size(1);
            int bias_width = bias.size(2);
            TORCH_CHECK(bias_channels == input_channels && bias_height == 1 && bias_width == 1, 
                        "Bias must have shape (channels, 1, 1)");

            int batch = input.size(0);
            int channels = input.size(1);
            int height = input.size(2);
            int width = input.size(3);
            int total_elements = batch * channels * height * width;

            auto output = torch::empty_like(input);

            const int block_size = 256;
            int num_blocks = (total_elements + block_size - 1) / block_size;

            fused_kernel<<<num_blocks, block_size>>>(
                input.data_ptr<float>(),
                bias.data_ptr<float>(),
                scaling_factor,
                output.data_ptr<float>(),
                batch, channels, height, width
            );

            return output;
        }
        """

        fused_kernel_header = """
        torch::Tensor fused_operation(torch::Tensor input, torch::Tensor bias, float scaling_factor);
        """

        # Load the fused CUDA kernel
        self.fused_op = load_inline(
            name="fused_op",
            cpp_sources=fused_kernel_header,
            cuda_sources=fused_kernel_source,
            functions=["fused_operation"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_ldflags=[""]
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_operation(x, self.bias, self.scaling_factor)
        return x

# Define the input and initialization functions
batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]