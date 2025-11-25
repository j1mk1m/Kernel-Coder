import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Inline CUDA code for fused element-wise operations
        fused_elementwise_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_elementwise_kernel(
            const float* input, 
            float scaling_factor,
            float hardtanh_min,
            float hardtanh_max,
            float* output,
            int size) 
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float temp = input[idx] * scaling_factor;
                temp = fmaxf(fminf(temp, hardtanh_max), hardtanh_min);
                // GELU approximation using sigmoid(1.702 * x)
                float inner = 1.702f * temp;
                float sigmoid = 1.0f / (1.0f + expf(-inner));
                output[idx] = temp * sigmoid;
            }
        }

        torch::Tensor fused_elementwise_cuda(torch::Tensor input, 
                                            float scaling_factor,
                                            float hardtanh_min,
                                            float hardtanh_max) {
            auto output = torch::empty_like(input);
            const int size = input.numel();
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_elementwise_kernel<<<num_blocks, block_size>>>(
                input.data_ptr<float>(),
                scaling_factor,
                hardtanh_min,
                hardtanh_max,
                output.data_ptr<float>(),
                size
            );

            return output;
        }
        """

        fused_elementwise_cpp_source = (
            "torch::Tensor fused_elementwise_cuda(torch::Tensor input, float scaling_factor, float hardtanh_min, float hardtanh_max);"
        )

        self.fused_elementwise = load_inline(
            name="fused_elementwise",
            cpp_sources=fused_elementwise_cpp_source,
            cuda_sources=fused_elementwise_source,
            functions=["fused_elementwise_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_elementwise.fused_elementwise_cuda(
            x.contiguous(),
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )
        return x

# Global variables as in the original code
batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]