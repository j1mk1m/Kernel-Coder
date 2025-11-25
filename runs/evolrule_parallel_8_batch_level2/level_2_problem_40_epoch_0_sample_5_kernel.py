import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

        # Define and compile the fused CUDA kernel for scaling and addition
        kernel_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        __global__ void fused_scale_add_kernel(float* output, const float* input, const float* residual, float scale, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = input[idx] * scale + residual[idx];
            }
        }

        torch::Tensor fused_scale_add(torch::Tensor input, torch::Tensor residual, float scale) {
            const int threads = 256;
            const int elements = input.numel();
            const int blocks = (elements + threads - 1) / threads;

            auto output = torch::empty_like(input);
            fused_scale_add_kernel<<<blocks, threads>>>(
                output.data_ptr<float>(),
                input.data_ptr<float>(),
                residual.data_ptr<float>(),
                scale,
                elements
            );
            return output;
        }
        """
        fused_scale_add = load_inline(
            name="fused_scale_add",
            cpp_sources="",
            cuda_sources=kernel_source,
            functions=["fused_scale_add"],
            verbose=True
        )
        self.fused_scale_add = fused_scale_add

    def forward(self, x):
        x = self.matmul(x)
        original_x = x.clone().detach()
        x = self.fused_scale_add.fused_scale_add(x, original_x, self.scaling_factor)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]