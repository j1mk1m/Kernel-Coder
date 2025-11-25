import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

        # Define and load the fused CUDA kernel
        fused_operations_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_operations_kernel(
            const float* input,
            const float* multiplier,
            float negative_slope,
            float* output,
            int num_elements,
            int channels,
            int height,
            int width
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_elements) return;

            int c = (idx / (height * width)) % channels;
            float val = input[idx];
            val *= multiplier[c];

            if (val < 0) val *= negative_slope;

            float inner = sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val);
            float tanh_inner = tanhf(inner);
            val = 0.5f * val * (1.0f + tanh_inner);

            output[idx] = val;
        }

        torch::Tensor fused_operations_cuda(
            torch::Tensor input,
            torch::Tensor multiplier,
            float negative_slope
        ) {
            input = input.contiguous();
            multiplier = multiplier.contiguous();
            auto output = torch::empty_like(input);

            int batch_size = input.size(0);
            int channels = input.size(1);
            int height = input.size(2);
            int width = input.size(3);
            int num_elements = input.numel();

            const int block_size = 256;
            int num_blocks = (num_elements + block_size - 1) / block_size;

            fused_operations_kernel<<<num_blocks, block_size>>>(
                input.data_ptr<float>(),
                multiplier.data_ptr<float>(),
                negative_slope,
                output.data_ptr<float>(),
                num_elements,
                channels,
                height,
                width
            );

            return output;
        }
        """

        fused_operations_cpp_header = (
            "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor multiplier, float negative_slope);"
        )

        self.fused_ops = load_inline(
            name="fused_ops",
            cpp_sources=[fused_operations_cpp_header],
            cuda_sources=[fused_operations_source],
            functions=["fused_operations_cuda"],
            verbose=True,
        )

    def forward(self, x):
        x = self.conv(x)
        negative_slope = self.leaky_relu.negative_slope
        x = self.fused_ops.fused_operations_cuda(x, self.multiplier, negative_slope)
        return x