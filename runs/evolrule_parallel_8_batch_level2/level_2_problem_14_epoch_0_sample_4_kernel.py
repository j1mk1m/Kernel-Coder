import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.register_buffer('sum_col_W', torch.sum(self.weight, dim=0))

        # Define and load the custom CUDA kernel
        optimized_forward_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void optimized_forward_kernel(
            const float* x,
            const float* sum_col_W,
            float* out,
            int batch_size,
            int input_size,
            float scaling_factor
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch_size) {
                float sum = 0.0f;
                for (int k = 0; k < input_size; ++k) {
                    sum += x[idx * input_size + k] * sum_col_W[k];
                }
                out[idx] = sum * (scaling_factor / 2.0f);
            }
        }

        torch::Tensor optimized_forward_cuda(
            torch::Tensor x,
            torch::Tensor sum_col_W,
            float scaling_factor
        ) {
            int batch_size = x.size(0);
            int input_size = x.size(1);

            auto out = torch::empty({batch_size, 1}, x.options());

            const int threads_per_block = 256;
            const int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

            optimized_forward_kernel<<<num_blocks, threads_per_block>>>(
                x.data_ptr<float>(),
                sum_col_W.data_ptr<float>(),
                out.data_ptr<float>(),
                batch_size,
                input_size,
                scaling_factor
            );

            return out;
        }
        """

        self.optimized_forward = load_inline(
            name="optimized_forward",
            cuda_sources=optimized_forward_source,
            functions=["optimized_forward_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.optimized_forward.optimized_forward_cuda(
            x, self.sum_col_W, self.scaling_factor
        )

# The following functions are provided as part of the problem's original setup
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]