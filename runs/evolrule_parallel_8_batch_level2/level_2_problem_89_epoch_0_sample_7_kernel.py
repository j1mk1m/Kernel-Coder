import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))  # Assuming subtraction is element-wise across channels

        # Define the custom CUDA kernel for fused operations
        fused_sub_swish_max_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_sub_swish_max(
            const float* input, 
            const float* subtract_param,
            float* output,
            int B, int C, int D, int H, int W
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= B * D * H * W) return;

            int b = idx / (D * H * W);
            int dw = idx % (D * H * W);
            int d = dw / (H * W);
            int hw = dw % (H * W);
            int h = hw / W;
            int w = hw % W;

            float max_val = -FLT_MAX;

            for (int c = 0; c < C; ++c) {
                int input_offset = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                float val = input[input_offset] - subtract_param[c];

                float sigmoid_val = 1.0f / (1.0f + expf(-val));
                float val_swish = val * sigmoid_val;

                if (val_swish > max_val) {
                    max_val = val_swish;
                }
            }

            int output_offset = b * D * H * W + d * H * W + h * W + w;
            output[output_offset] = max_val;
        }

        torch::Tensor fused_sub_swish_max_cuda(torch::Tensor input, torch::Tensor subtract_param) {
            int B = input.size(0);
            int C = input.size(1);
            int D = input.size(2);
            int H = input.size(3);
            int W = input.size(4);

            auto output = torch::empty({B, D, H, W}, input.options());

            int total_threads = B * D * H * W;
            int threads_per_block = 256;
            int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

            fused_sub_swish_max<<<blocks_per_grid, threads_per_block>>>(
                input.data_ptr<float>(),
                subtract_param.data_ptr<float>(),
                output.data_ptr<float>(),
                B, C, D, H, W
            );

            return output;
        }
        """

        fused_sub_swish_max_cpp_source = """
        torch::Tensor fused_sub_swish_max_cuda(torch::Tensor input, torch::Tensor subtract_param);
        """

        self.fused_sub_swish_max = load_inline(
            name="fused_sub_swish_max",
            cpp_sources=fused_sub_swish_max_cpp_source,
            cuda_sources=fused_sub_swish_max_source,
            functions=["fused_sub_swish_max_cuda"],
            verbose=True,
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = torch.softmax(x, dim=1)
        # Pass the subtract parameter as is
        x = self.fused_sub_swish_max.fused_sub_swish_max_cuda(x, self.subtract)
        return x