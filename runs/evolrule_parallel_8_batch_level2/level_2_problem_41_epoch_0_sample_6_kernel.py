import torch
import torch.nn as nn
import math

from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize GEMM parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize with the same default initializations as nn.Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize batch norm parameters
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.eps = 1e-5  # same as default for BatchNorm1d
        
        # Define and load the fused kernel
        fused_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void fused_operations_kernel(
            const float* input,
            float* output,
            const float* running_mean,
            const float* running_var,
            const float* gamma,
            const float* beta,
            float eps,
            int batch_size,
            int out_features
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_features) return;

            int k = idx % out_features;
            int i = idx / out_features;

            float x_val = input[idx];

            // BatchNorm computation
            float mean = running_mean[k];
            float var = running_var[k];
            float inv_std = 1.0f / sqrtf(var + eps);
            float normalized = (x_val - mean) * inv_std;
            float bn_out = normalized * gamma[k] + beta[k];

            // GELU computation using erf
            float scaled = bn_out;
            float sqrt2 = 1.41421356237f;
            float y = scaled / sqrt2;
            float erf_y = erff(y); // erff is for float in CUDA
            float gelu_val = scaled * 0.5f * (1.0f + erf_y);

            // ReLU
            float relu_val = fmaxf(gelu_val, 0.0f); // redundant but required per original code

            output[idx] = relu_val;
        }

        torch::Tensor fused_operations_cuda(torch::Tensor input,
                                           torch::Tensor running_mean,
                                           torch::Tensor running_var,
                                           torch::Tensor gamma,
                                           torch::Tensor beta,
                                           float eps) {
            int batch_size = input.size(0);
            int out_features = input.size(1);
            auto output = torch::empty_like(input);

            const int block_size = 256;
            int total_elements = batch_size * out_features;
            int num_blocks = (total_elements + block_size - 1) / block_size;

            fused_operations_kernel<<<num_blocks, block_size>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                gamma.data_ptr<float>(),
                beta.data_ptr<float>(),
                eps,
                batch_size,
                out_features
            );

            return output;
        }
        """
        fused_cpp_source = """
        torch::Tensor fused_operations_cuda(torch::Tensor input,
                                           torch::Tensor running_mean,
                                           torch::Tensor running_var,
                                           torch::Tensor gamma,
                                           torch::Tensor beta,
                                           float eps);
        """
        
        fused_ops = load_inline(
            name="fused_ops",
            cpp_sources=fused_cpp_source,
            cuda_sources=fused_source,
            functions=["fused_operations_cuda"],
            verbose=True,
            extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
            extra_cuda_cflags=["-lineinfo", "-arch=sm_80"]
        )
        self.fused_operations = fused_ops.fused_operations_cuda

    def forward(self, x):
        # GEMM step using torch's linear
        x = F.linear(x, self.weight, self.bias)
        # Apply fused operations
        return self.fused_operations(
            x,
            self.running_mean,
            self.running_var,
            self.gamma,
            self.beta,
            self.eps
        )