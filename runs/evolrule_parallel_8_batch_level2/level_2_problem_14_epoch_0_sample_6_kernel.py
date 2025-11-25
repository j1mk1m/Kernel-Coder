import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size).cuda())
        self.scaling_factor = scaling_factor

        # Define the CUDA kernel for the fused operation
        elementwise_sum_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_elementwise_sum(
            const float* x,
            const float* S,
            float* out,
            int batch_size,
            int input_size,
            float scale
        ) {
            int batch_idx = blockIdx.x;
            if (batch_idx >= batch_size) return;

            extern __shared__ float sdata[];
            float sum = 0.0f;

            for (int k = threadIdx.x; k < input_size; k += blockDim.x) {
                sum += x[batch_idx * input_size + k] * S[k];
            }

            sdata[threadIdx.x] = sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    sdata[threadIdx.x] += sdata[threadIdx.x + s];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                out[batch_idx] = sdata[0] * scale;
            }
        }

        torch::Tensor fused_elementwise_sum_cuda(
            torch::Tensor x,
            torch::Tensor S,
            float scale
        ) {
            const int batch_size = x.size(0);
            const int input_size = x.size(1);
            auto out = torch::empty({batch_size, 1}, x.options());

            const int block_size = 256;
            dim3 grid(batch_size);
            dim3 block(block_size);

            fused_elementwise_sum<<<grid, block, block_size * sizeof(float)>>>(
                x.data_ptr<float>(),
                S.data_ptr<float>(),
                out.data_ptr<float>(),
                batch_size,
                input_size,
                scale
            );

            return out;
        }
        """

        elementwise_sum_cpp_source = """
        torch::Tensor fused_elementwise_sum_cuda(
            torch::Tensor x,
            torch::Tensor S,
            float scale
        );
        """

        # Compile the CUDA kernel
        self.fused_elementwise_sum = load_inline(
            name="fused_elementwise_sum",
            cpp_sources=elementwise_sum_cpp_source,
            cuda_sources=elementwise_sum_source,
            functions=["fused_elementwise_sum_cuda"],
            verbose=True,
        )

    def forward(self, x):
        # Compute S = sum(weight, dim=0)
        S = torch.sum(self.weight, dim=0)
        # Compute the scaling factor divided by 2
        scale = self.scaling_factor / 2.0
        # Execute the fused kernel
        out = self.fused_elementwise_sum.fused_elementwise_sum_cuda(x, S, scale)
        return out

# Ensure the variables are as per original code
batch_size = 1024  
input_size = 8192  
hidden_size = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]