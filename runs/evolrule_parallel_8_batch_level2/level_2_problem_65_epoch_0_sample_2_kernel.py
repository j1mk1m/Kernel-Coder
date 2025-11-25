import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

        # Define fused sigmoid + sum CUDA kernel
        sigmoid_sum_source = """
        #include <torch/extension.h>
        #include <math.h>
        #include <cuda_runtime.h>

        __global__ void sigmoid_sum_kernel(
            const float* input,
            float* output,
            int B,
            int C,
            int H,
            int W
        ) {
            int batch_idx = blockIdx.x;
            int elements_per_batch = C * H * W;
            int tid = threadIdx.x;
            int num_threads = blockDim.x;

            float sum = 0.0f;

            for (int i = tid; i < elements_per_batch; i += num_threads) {
                int idx = batch_idx * C * H * W + i;
                float val = input[idx];
                val = 1.0f / (1.0f + expf(-val));
                sum += val;
            }

            // Block reduction using shared memory
            __shared__ float shared_mem[256];
            shared_mem[threadIdx.x] = sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                output[batch_idx] = shared_mem[0];
            }
        }

        torch::Tensor sigmoid_sum_cuda(torch::Tensor input) {
            auto B = input.size(0);
            auto C = input.size(1);
            auto H = input.size(2);
            auto W = input.size(3);

            auto output = torch::zeros({B}, torch::dtype(input.dtype()).device(input.device()));

            const int threads_per_block = 256;
            const int blocks_per_grid = B;

            sigmoid_sum_kernel<<<blocks_per_grid, threads_per_block>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                B, C, H, W
            );

            // Check for CUDA errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
            }

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA device sync failed: " + std::string(cudaGetErrorString(err)));
            }

            return output;
        }
        """

        sigmoid_sum_cpp_source = "torch::Tensor sigmoid_sum_cuda(torch::Tensor input);"

        # Compile CUDA kernel
        self.sigmoid_sum = load_inline(
            name="sigmoid_sum",
            cpp_sources=sigmoid_sum_cpp_source,
            cuda_sources=sigmoid_sum_source,
            functions=["sigmoid_sum_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = self.sigmoid_sum.sigmoid_sum_cuda(x)
        return x