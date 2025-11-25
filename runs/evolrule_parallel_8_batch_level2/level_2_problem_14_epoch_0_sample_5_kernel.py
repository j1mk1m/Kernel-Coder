import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

        # Define the fused CUDA kernel for dot product and scaling
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void fused_dot_scale_kernel(
            const scalar_t* __restrict__ x,
            const scalar_t* __restrict__ summed_weights,
            scalar_t* __restrict__ out,
            int batch_size,
            int input_size,
            scalar_t scale_factor) {

            int batch_idx = blockIdx.x;

            scalar_t sum = 0.0;
            for (int i = 0; i < input_size; ++i) {
                sum += x[batch_idx * input_size + i] * summed_weights[i];
            }
            out[batch_idx] = sum * scale_factor;
        }

        torch::Tensor fused_dot_scale(
            torch::Tensor x,
            torch::Tensor summed_weights,
            float scaling_factor) {

            const int batch_size = x.size(0);
            const int input_size = x.size(1);
            const auto scale = scaling_factor / 2.0;

            auto out = torch::empty({batch_size, 1}, x.options());

            const int threads_per_block = 256;
            const int blocks_per_grid = batch_size;

            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_dot_scale", ([&] {
                fused_dot_scale_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                    x.data<scalar_t>(),
                    summed_weights.data<scalar_t>(),
                    out.data<scalar_t>(),
                    batch_size,
                    input_size,
                    scale
                );
            }));

            return out;
        }
        """

        # Compile the kernel
        self.fused_dot_scale = load_inline(
            name="fused_dot_scale",
            cpp_sources="",
            cuda_sources=fused_kernel_source,
            functions=["fused_dot_scale"],
            verbose=True
        )

    def forward(self, x):
        # Ensure input is on GPU
        x = x.cuda()
        # Precompute summed weights over the hidden dimension
        summed_weights = self.weight.sum(0)
        # Apply the fused kernel
        out = self.fused_dot_scale.fused_dot_scale(x, summed_weights, self.scaling_factor)
        return out

# Define input generation functions as per the original problem
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]