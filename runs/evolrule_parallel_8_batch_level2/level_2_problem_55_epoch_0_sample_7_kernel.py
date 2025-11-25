import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Define the custom CUDA kernel for max_pool, sum, and scaling
        fused_max_sum_scale_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_max_sum_scale(
            const float* x_linear,
            float* output,
            int batch_size,
            int out_features,
            int kernel_size,
            float scale_factor
        ) {
            int batch_idx = blockIdx.x;
            if (batch_idx >= batch_size) return;

            float sum = 0.0f;

            for (int i = 0; i < out_features; i += kernel_size) {
                float a = x_linear[batch_idx * out_features + i];
                float b = x_linear[batch_idx * out_features + i + 1];
                sum += fmaxf(a, b);
            }

            output[batch_idx] = sum * scale_factor;
        }

        torch::Tensor fused_max_sum_scale_cuda(torch::Tensor x_linear, 
                                              int out_features,
                                              int kernel_size,
                                              float scale_factor) {
            int batch_size = x_linear.size(0);
            auto output = torch::empty({batch_size}, x_linear.options());

            const int threads_per_block = 256;
            const int blocks_per_grid = batch_size;

            fused_max_sum_scale<<<blocks_per_grid, threads_per_block>>>(
                x_linear.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                out_features,
                kernel_size,
                scale_factor
            );

            return output;
        }
        """

        fused_max_sum_scale_cpp = """
        torch::Tensor fused_max_sum_scale_cuda(torch::Tensor x_linear, int out_features, int kernel_size, float scale_factor);
        """

        # Load the CUDA kernel
        self.fused_max_sum_scale = load_inline(
            name="fused_max_sum_scale",
            cpp_sources=fused_max_sum_scale_cpp,
            cuda_sources=fused_max_sum_scale_source,
            functions=["fused_max_sum_scale_cuda"],
            verbose=True
        )

    def forward(self, x):
        x_linear = self.matmul(x)
        out_features = x_linear.size(1)
        output = self.fused_max_sum_scale.fused_max_sum_scale_cuda(
            x_linear, out_features, self.kernel_size, self.scale_factor
        )
        return output