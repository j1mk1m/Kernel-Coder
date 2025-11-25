import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Define custom CUDA kernels for min and fused tanh operations
        self.min_tanh_tanh = load_inline(
            name="min_tanh_tanh",
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __device__ scalar_t fused_tanh_tanh(scalar_t x) {{
                // Compute tanh(tanh(x)) in one step to save computation
                scalar_t t = tanh(x);
                return tanh(t);
            }}

            __global__ void min_tanh_tanh_kernel(
                const scalar_t* __restrict__ input,
                scalar_t* __restrict__ output,
                int batch_size,
                int out_channels,
                int height,
                int width,
                int channels_dim) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= batch_size * height * width) return;

                // Compute min along the channel dimension (dim=1)
                scalar_t min_val = input[idx * channels_dim];
                for (int c = 1; c < channels_dim; ++c) {{
                    scalar_t val = input[idx * channels_dim + c];
                    if (val < min_val) min_val = val;
                }}

                // Apply fused tanh twice
                output[idx] = fused_tanh_tanh(min_val);
            }}

            torch::Tensor min_tanh_tanh_cuda(torch::Tensor input) {{
                const int batch_size = input.size(0);
                const int channels = input.size(1);
                const int height = input.size(2);
                const int width = input.size(3);
                const int spatial_size = batch_size * height * width;
                const int channels_dim = channels;

                auto output = torch::empty({{batch_size, 1, height, width}}, 
                    input.options());
                
                const int threads = 256;
                const int blocks = (spatial_size + threads - 1) / threads;

                // Launch kernel
                min_tanh_tanh_kernel<<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    height,
                    width,
                    channels_dim);
                
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                return output;
            }}
            """,
            functions=["min_tanh_tanh_cuda"],
            extra_cuda_cflags=['-arch=sm_75'],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x)
        # Perform min over channel dimension and apply both tanh operations in one kernel
        x = self.min_tanh_tanh.min_tanh_tanh_cuda(x)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 16
    height = width = 256
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [16, 64, 3]

# Define CUDA error checking macro
CUDA_CHECK = lambda x: x