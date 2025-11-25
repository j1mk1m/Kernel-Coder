import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

        # Define CUDA kernels for fused operations
        # Kernel 1: Mean over depth, add bias
        mean_add_bias_source = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void mean_add_bias_kernel(
            const scalar_t* input,
            scalar_t* output,
            const scalar_t* bias,
            int B, int C, int D, int H, int W) {{
            int b = blockIdx.x;
            int c = blockIdx.y;
            int h = threadIdx.x;
            int w = threadIdx.y;

            if (h >= H || w >= W) return;

            scalar_t sum = 0.0;
            for (int d = 0; d < D; ++d) {{
                int idx = b * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
                sum += input[idx];
            }}
            scalar_t mean_val = sum / D;
            int bias_idx = c;  // Bias is [1, C, 1, 1, 1]
            scalar_t bias_val = bias[bias_idx];
            output[b * C * H * W + c * H * W + h * W + w] = mean_val + bias_val;
        }}

        torch::Tensor mean_add_bias_cuda(torch::Tensor input, torch::Tensor bias, int B, int C, int D, int H, int W) {{
            auto output = torch::empty({{B, C, 1, H, W}}, input.options());
            dim3 threads(H, W);
            dim3 blocks(B, C);
            mean_add_bias_kernel<float><<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                bias.data_ptr<float>(),
                B, C, D, H, W
            );
            return output;
        }}
        """
        # Compile mean_add_bias kernel
        mean_add_bias = load_inline(
            name="mean_add_bias",
            cpp_sources=f"torch::Tensor mean_add_bias_cuda(torch::Tensor input, torch::Tensor bias, int B, int C, int D, int H, int W);",
            cuda_sources=mean_add_bias_source,
            functions=["mean_add_bias_cuda"],
            verbose=True
        )
        self.mean_add_bias_cuda = mean_add_bias.mean_add_bias_cuda

        # Kernel 2: Softmax over channels, tanh, scaling
        softmax_tanh_scale_source = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cmath>

        template <typename scalar_t>
        __global__ void softmax_tanh_scale_kernel(
            const scalar_t* input,
            scalar_t* output,
            float scaling_factor,
            int B, int C, int H, int W) {{
            int b = blockIdx.x;
            int h = blockIdx.y;
            int w = blockIdx.z;
            int c = threadIdx.x;

            extern __shared__ float shared[];
            scalar_t* x_vals = (scalar_t*)shared;
            scalar_t* exp_vals = x_vals + C;

            int input_offset = b * C * H * W + c * H * W + h * W + w;
            scalar_t x = input[input_offset];

            x_vals[threadIdx.x] = x;
            __syncthreads();

            scalar_t max_val = -INFINITY;
            for (int i = 0; i < C; ++i) {{
                if (x_vals[i] > max_val) max_val = x_vals[i];
            }}
            __syncthreads();

            scalar_t numerator = expf(x - max_val);
            exp_vals[threadIdx.x] = numerator;
            __syncthreads();

            scalar_t sum = 0.0f;
            for (int i = 0; i < C; ++i) sum += exp_vals[i];
            __syncthreads();

            scalar_t softmax_val = numerator / sum;
            scalar_t result = tanhf(softmax_val) * scaling_factor;

            output[input_offset] = result;
        }}

        torch::Tensor softmax_tanh_scale_cuda(torch::Tensor input, float scaling_factor, int B, int C, int H, int W) {{
            auto output = torch::empty_like(input);
            dim3 threads(C);
            dim3 blocks(B, H, W);
            int sm_size = 2 * C * sizeof(float);
            softmax_tanh_scale_kernel<float><<<blocks, threads, sm_size>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                scaling_factor,
                B, C, H, W
            );
            return output;
        }}
        """
        # Compile softmax_tanh_scale kernel
        softmax_tanh_scale = load_inline(
            name="softmax_tanh_scale",
            cpp_sources=f"torch::Tensor softmax_tanh_scale_cuda(torch::Tensor input, float scaling_factor, int B, int C, int H, int W);",
            cuda_sources=softmax_tanh_scale_source,
            functions=["softmax_tanh_scale_cuda"],
            verbose=True
        )
        self.softmax_tanh_scale_cuda = softmax_tanh_scale.softmax_tanh_scale_cuda

    def forward(self, x):
        # Convolution transpose (PyTorch op remains)
        x = self.conv_transpose(x)
        
        # Get tensor dimensions
        B, C, D, H, W = x.shape
        
        # Fused mean_add_bias (CUDA kernel)
        x = self.mean_add_bias_cuda(x, self.bias, B, C, D, H, W)
        
        # Fused softmax_tanh_scale (CUDA kernel)
        x = self.softmax_tanh_scale_cuda(x, self.scaling_factor, B, C, H, W)
        
        return x