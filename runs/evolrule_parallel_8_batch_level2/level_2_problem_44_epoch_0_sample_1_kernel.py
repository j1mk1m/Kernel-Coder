import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

        # Define the fused kernel
        fused_multiply_and_pool_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename T>
        __global__ void fused_multiply_and_pool_kernel(
            const T* input,
            T multiplier,
            T* output,
            int batch_size,
            int channels,
            int height,
            int width
        ) {
            extern __shared__ T partial_sums[];
            int b = blockIdx.y;
            int c = blockIdx.x;
            
            if (b >= batch_size || c >= channels) return;
            
            int tid = threadIdx.x;
            
            T sum = 0.0;
            for (int idx = tid; idx < height * width; idx += blockDim.x) {
                int h = idx / width;
                int w = idx % width;
                int in_idx = ((b * channels + c) * height + h) * width + w;
                sum += input[in_idx] * multiplier;
            }
            
            partial_sums[tid] = sum;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    partial_sums[tid] += partial_sums[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                output[(b * channels + c)] = partial_sums[0] / (height * width);
            }
        }

        torch::Tensor fused_multiply_and_pool_cuda(torch::Tensor input, float multiplier) {
            auto batch_size = input.size(0);
            auto channels = input.size(1);
            auto height = input.size(2);
            auto width = input.size(3);
            
            auto output = torch::empty({batch_size, channels, 1, 1}, input.options());
            
            const int block_size = 256;
            dim3 block(block_size);
            dim3 grid(channels, batch_size);
            
            int shared_mem = block_size * sizeof(float);
            
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_multiply_and_pool_cuda", ([&] {
                fused_multiply_and_pool_kernel<scalar_t><<<grid, block, shared_mem>>>(
                    input.data_ptr<scalar_t>(),
                    multiplier,
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    height,
                    width
                );
            }));
            
            cudaDeviceSynchronize();
            return output;
        }
        """

        fused_multiply_and_pool_cpp_source = """
        torch::Tensor fused_multiply_and_pool_cuda(torch::Tensor input, float multiplier);
        """

        self.fused_multiply_and_pool = load_inline(
            name="fused_multiply_and_pool",
            cpp_sources=fused_multiply_and_pool_cpp_source,
            cuda_sources=fused_multiply_and_pool_source,
            functions=["fused_multiply_and_pool_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_multiply_and_pool.fused_multiply_and_pool_cuda(x, self.multiplier)
        return x