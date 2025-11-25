import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

        # Define the fused CUDA kernel
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_linear_srm_relu_kernel(
            const float* input,
            const float* weight,
            const float* bias,
            float subtract_val,
            float multiply_val,
            float* output,
            int batch_size,
            int in_features,
            int out_features
        ) {
            int batch_idx = blockIdx.x;
            int out_elem = threadIdx.x;

            if (out_elem < out_features) {
                float sum = 0.0;
                for (int in_idx = 0; in_idx < in_features; ++in_idx) {
                    sum += input[batch_idx * in_features + in_idx] * 
                           weight[out_elem * in_features + in_idx];
                }
                sum += bias[out_elem];  // Add bias
                sum -= subtract_val;    // Subtract scalar
                sum *= multiply_val;    // Multiply scalar
                output[batch_idx * out_features + out_elem] = 
                    sum > 0.0 ? sum : 0.0;  // ReLU
            }
        }

        torch::Tensor fused_linear_srm_relu(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            float subtract_val,
            float multiply_val
        ) {
            const int batch_size = input.size(0);
            const int out_features = weight.size(0);
            auto output = torch::empty({batch_size, out_features}, 
                                      dtype: input.dtype(), 
                                      device: input.device());

            const int threads_per_block = 256;  // Adjust as needed
            const dim3 blocks(batch_size);
            const dim3 threads(out_features);

            fused_linear_srm_relu_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                subtract_val,
                multiply_val,
                output.data_ptr<float>(),
                batch_size,
                input.size(1),
                out_features
            );

            return output;
        }
        """
        fused_kernel_cpp_source = (
            "torch::Tensor fused_linear_srm_relu("
            "torch::Tensor input, "
            "torch::Tensor weight, "
            "torch::Tensor bias, "
            "float subtract_val, "
            "float multiply_val);"
        )

        # Compile the fused kernel
        self.fused_linear_srm_relu = load_inline(
            name="fused_linear_srm_relu",
            cpp_sources=fused_kernel_cpp_source,
            cuda_sources=fused_kernel_source,
            functions=["fused_linear_srm_relu"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_ldflags=[""]
        )

    def forward(self, x):
        # Extract parameters from linear layer
        weight = self.linear.weight
        bias = self.linear.bias

        # Launch fused kernel
        output = self.fused_linear_srm_relu.fused_linear_srm_relu(
            x,
            weight,
            bias,
            self.subtract_value,
            self.multiply_value
        )

        return output

def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    in_features = 8192
    out_features = 8192
    subtract_value = 2.0
    multiply_value = 1.5
    return [in_features, out_features, subtract_value, multiply_value]