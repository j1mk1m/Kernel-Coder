import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        # Define the fused CUDA kernel for min + two tanh operations
        fused_min_tanh_tanh_source = """
        #include <torch/extension.h>
        #include <math.h>

        __global__ void fused_min_tanh_tanh_kernel(
            const float* input, float* output,
            int batch_size, int out_channels, int H, int W) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * H * W) return;

            int b = idx / (H * W);
            int rem = idx % (H * W);
            int x = rem / W;
            int y = rem % W;

            float min_val = INFINITY;
            for (int c = 0; c < out_channels; c++) {
                int input_offset = b * out_channels * H * W + c * H * W + x * W + y;
                float val = input[input_offset];
                if (val < min_val) {
                    min_val = val;
                }
            }

            float res = tanhf(min_val);
            res = tanhf(res);

            int output_offset = b * H * W + x * W + y;
            output[output_offset] = res;
        }

        torch::Tensor fused_min_tanh_tanh_cuda(torch::Tensor input) {
            int batch_size = input.size(0);
            int out_channels = input.size(1);
            int H = input.size(2);
            int W = input.size(3);

            auto output = torch::empty({batch_size, 1, H, W}, input.options());

            const int threads_per_block = 256;
            const int blocks_per_grid = (batch_size * H * W + threads_per_block - 1) / threads_per_block;

            fused_min_tanh_tanh_kernel<<<blocks_per_grid, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, out_channels, H, W
            );

            return output;
        }
        """

        fused_min_tanh_tanh_cpp = "torch::Tensor fused_min_tanh_tanh_cuda(torch::Tensor input);"
        self.fused_min_tanh_tanh = load_inline(
            name="fused_min_tanh_tanh",
            cpp_sources=fused_min_tanh_tanh_cpp,
            cuda_sources=fused_min_tanh_tanh_source,
            functions=["fused_min_tanh_tanh_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_min_tanh_tanh.fused_min_tanh_tanh_cuda(x)
        return x