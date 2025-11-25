import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

        # Define and compile the fused activation CUDA kernel
        elementwise_fused_source = """
        #include <torch/extension.h>
        #include <math.h>
        #include <cuda_runtime.h>

        __global__ void fused_activation_kernel(
            const float* matmul_data,
            const float* add_value,
            float* out,
            int batch_size,
            int out_features
        ) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= batch_size * out_features) return;

            int batch = index / out_features;
            int feature = index % out_features;

            float x = matmul_data[index] + add_value[feature];

            // Apply Swish
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            x *= sigmoid_x;

            // Apply tanh
            x = tanhf(x);

            // Apply GELU approximation
            const float sqrt_2_over_pi = sqrt(2.0f / M_PI);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
            float tanh_val = tanhf(inner);
            x = 0.5f * x * (1.0f + tanh_val);

            // Apply Hardtanh
            if (x < -1.0f) x = -1.0f;
            else if (x > 1.0f) x = 1.0f;

            out[index] = x;
        }

        torch::Tensor fused_activation_cuda(torch::Tensor matmul_out, torch::Tensor add_value) {
            int batch_size = matmul_out.size(0);
            int out_features = matmul_out.size(1);
            assert(add_value.size(0) == out_features);

            auto output = torch::empty_like(matmul_out);
            int total_elements = batch_size * out_features;

            const int block_size = 256;
            int num_blocks = (total_elements + block_size - 1) / block_size;

            fused_activation_kernel<<<num_blocks, block_size>>>(
                matmul_out.data_ptr<float>(),
                add_value.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                out_features
            );

            return output;
        }
        """

        elementwise_fused_cpp_header = "torch::Tensor fused_activation_cuda(torch::Tensor, torch::Tensor);"
        fused_activation = load_inline(
            name="fused_activation",
            cpp_sources=elementwise_fused_cpp_header,
            cuda_sources=elementwise_fused_source,
            functions=["fused_activation_cuda"],
            verbose=False
        )

        self.fused_activation = fused_activation

    def forward(self, x):
        matmul_out = self.matmul(x)
        return self.fused_activation.fused_activation_cuda(matmul_out, self.add_value)