import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.compute_partial_sums_launch = self._load_cuda_kernel('compute_partial_sums')
        self.reduce_partial_sums_launch = self._load_cuda_kernel('reduce_partial_sums')
        self.divide_by_norm_launch = self._load_cuda_kernel('divide_by_norm')

    def _load_cuda_kernel(self, kernel_name):
        if kernel_name == 'compute_partial_sums':
            cuda_source = """
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            __global__ void compute_partial_sums(const float* x, float* partial_sums, int total_elements, int num_blocks) {
                extern __shared__ float shared[];
                int tid = threadIdx.x;
                int bid = blockIdx.x;

                shared[tid] = 0.0f;
                __syncthreads();

                for (int i = bid * blockDim.x + tid; i < total_elements; i += num_blocks * blockDim.x) {
                    float val = x[i];
                    shared[tid] += val * val;
                }

                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    __syncthreads();
                    if (tid < s) {
                        shared[tid] += shared[tid + s];
                    }
                }

                __syncthreads();

                if (tid == 0) {
                    partial_sums[bid] = shared[0];
                }
            }
            """
            cpp_source = """
            torch::Tensor compute_partial_sums_launch(torch::Tensor x, torch::Tensor partial_sums, int total_elements, int num_blocks) {
                const int block_size = 256;
                dim3 threads(block_size);
                dim3 blocks(num_blocks);

                compute_partial_sums<<<blocks, threads, block_size * sizeof(float)>>>(
                    x.data_ptr<float>(),
                    partial_sums.data_ptr<float>(),
                    total_elements,
                    num_blocks
                );
                return partial_sums;
            }
            """
        elif kernel_name == 'reduce_partial_sums':
            cuda_source = """
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            __global__ void reduce_partial_sums(float* input, int n, float* output) {
                extern __shared__ float shared[];
                int tid = threadIdx.x;
                float sum = 0.0f;

                for (int i = tid; i < n; i += blockDim.x) {
                    sum += input[i];
                }

                shared[tid] = sum;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        shared[tid] += shared[tid + s];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    output[0] = shared[0];
                }
            }
            """
            cpp_source = """
            torch::Tensor reduce_partial_sums_launch(torch::Tensor input, torch::Tensor output) {
                int num_partial = input.sizes()[0];
                const int block_size = 1024;
                dim3 threads(block_size);
                dim3 blocks(1);
                reduce_partial_sums<<<blocks, threads, block_size * sizeof(float)>>>(
                    input.data_ptr<float>(),
                    num_partial,
                    output.data_ptr<float>()
                );
                return output;
            }
            """
        elif kernel_name == 'divide_by_norm':
            cuda_source = """
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            __global__ void divide_by_norm(const float* x, float norm, float* out, int total_elements) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < total_elements) {
                    out[idx] = x[idx] / norm;
                }
            }
            """
            cpp_source = """
            torch::Tensor divide_by_norm_launch(torch::Tensor x, float norm, torch::Tensor out) {
                const int total_elements = x.numel();
                const int block_size = 256;
                const int num_blocks = (total_elements + block_size - 1) / block_size;
                dim3 threads(block_size);
                dim3 blocks(num_blocks);

                divide_by_norm<<<blocks, threads>>>(
                    x.data_ptr<float>(),
                    norm,
                    out.data_ptr<float>(),
                    total_elements
                );
                return out;
            }
            """
        else:
            raise ValueError(f"Unknown kernel name: {kernel_name}")

        module = load_inline(
            name=f"custom_{kernel_name}_kernel",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=[f"{kernel_name}_launch"],
            verbose=True
        )
        return getattr(module, f"{kernel_name}_launch")

    def forward(self, x):
        total_elements = x.numel()
        num_blocks = 4096
        block_size = 256

        # Step 1: Compute partial sums of squares
        partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=x.device)
        partial_sums = self.compute_partial_sums_launch(x, partial_sums, total_elements, num_blocks)

        # Step 2: Reduce partial_sums to total_sum
        total_sum = torch.zeros(1, dtype=torch.float32, device=x.device)
        total_sum = self.reduce_partial_sums_launch(partial_sums, total_sum)

        # Compute the norm on CPU
        norm = torch.sqrt(total_sum).item()

        # Step 3: Divide each element by norm
        out = torch.empty_like(x)
        out = self.divide_by_norm_launch(x, norm, out)

        return out