import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Define the fused kernel code
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <math.h>

        __global__ void fused_kernel(const float* x_data,
                                    const float* scale_data,
                                    const float* shift_data,
                                    float* out_data,
                                    int batch_size,
                                    int out_features) {
            const float sqrt_2_over_pi = 0.7978845608f;
            const float a = 0.044715f;
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_features)
                return;

            int c = idx % out_features;

            float temp = x_data[idx] * scale_data[c] + shift_data[c];

            float x_pow3 = temp * temp * temp;
            float tanh_arg = sqrt_2_over_pi * (temp + a * x_pow3);
            float tanh_val = tanhf(tanh_arg);
            float gelu_val = 0.5f * temp * (1.0f + tanh_val);

            out_data[idx] = fmaxf(gelu_val, 0.0f);
        }

        torch::Tensor fused_batchnorm_gelu_relu_cuda(torch::Tensor x,
                                                    torch::Tensor scale,
                                                    torch::Tensor shift) {
            int batch_size = x.size(0);
            int out_features = x.size(1);
            auto out = torch::empty_like(x);

            const int threads_per_block = 256;
            const int num_blocks = (batch_size * out_features + threads_per_block - 1) / threads_per_block;

            fused_kernel<<<num_blocks, threads_per_block>>>(
                x.data_ptr<float>(),
                scale.data_ptr<float>(),
                shift.data_ptr<float>(),
                out.data_ptr<float>(),
                batch_size,
                out_features
            );

            return out;
        }
        """
        
        fused_kernel_cpp_source = """
        torch::Tensor fused_batchnorm_gelu_relu_cuda(torch::Tensor x,
                                                    torch::Tensor scale,
                                                    torch::Tensor shift);
        """
        
        # Compile the fused kernel
        self.fused = load_inline(
            name="fused_batchnorm_gelu_relu",
            cpp_sources=fused_kernel_cpp_source,
            cuda_sources=fused_kernel_source,
            functions=["fused_batchnorm_gelu_relu_cuda"],
            verbose=False
        )

    def forward(self, x):
        x = self.gemm(x)
        
        # Extract batch norm parameters
        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        eps = self.batch_norm.eps

        # Precompute scale and shift
        scale = gamma / torch.sqrt(running_var + eps)
        shift = beta - (running_mean * gamma) / torch.sqrt(running_var + eps)

        # Apply fused kernel
        x = self.fused.fused_batchnorm_gelu_relu_cuda(x, scale, shift)
        return x

# The original get_inputs and get_init_inputs remain unchanged
batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]