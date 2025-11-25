import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

        custom_cumsum_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void custom_cumsum_kernel(
            scalar_t* out,
            const scalar_t* x,
            const unsigned char* mask,
            int dim_size,
            int batch_size
        ) {
            extern __shared__ float shared_data[];
            const int tid = threadIdx.x;
            const int num_threads = blockDim.x;
            const int batch_index = blockIdx.x;

            const int K = 32;
            const int start = tid * K;
            const int end = min(start + K, dim_size);

            scalar_t segment_values[K];
            scalar_t local_cumulative[K];
            for (int i = start; i < end; ++i) {
                int global_idx = batch_index * dim_size + i;
                scalar_t val = x[global_idx] * static_cast<scalar_t>(mask[global_idx] ? 1 : 0);
                int local_idx = i - start;
                segment_values[local_idx] = val;
            }

            if (start < end) {
                local_cumulative[0] = segment_values[0];
                for (int j = 1; j < (end - start); ++j) {
                    local_cumulative[j] = local_cumulative[j - 1] + segment_values[j];
                }
            }

            scalar_t segment_sum = 0.0;
            for (int j = 0; j < (end - start); ++j) {
                segment_sum += segment_values[j];
            }

            if (tid < num_threads) {
                shared_data[tid] = static_cast<float>(segment_sum);
            } else {
                shared_data[tid] = 0.0f;
            }
            __syncthreads();

            for (int d = 1; d < num_threads; d *= 2) {
                int ai = 2 * d * tid;
                int aj = ai + d;
                if (aj < num_threads) {
                    shared_data[ai] += shared_data[aj];
                }
                __syncthreads();
            }

            if (tid == 0) {
                shared_data[num_threads - 1] = 0.0f;
            }
            __syncthreads();

            for (int d = num_threads >> 1; d > 0; d >>= 1) {
                int ai = 2 * d * tid;
                int aj = ai + d;
                if (aj < num_threads) {
                    shared_data[aj] += shared_data[ai];
                }
                __syncthreads();
            }

            float prefix = shared_data[tid];

            for (int i = start; i < end; ++i) {
                int local_idx = i - start;
                int global_idx = batch_index * dim_size + i;
                out[global_idx] = static_cast<scalar_t>(prefix) + local_cumulative[local_idx];
            }
        }

        at::Tensor custom_cumsum_cuda(at::Tensor x, at::Tensor mask, int dim_size, int batch_size) {
            const int threads_per_block = 1024;
            const dim3 blocks(batch_size);
            const dim3 threads(threads_per_block);
            const size_t shared_size = threads_per_block * sizeof(float);

            auto output = at::empty_like(x);

            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "custom_cumsum_cuda", ([&] {
                custom_cumsum_kernel<scalar_t><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
                    output.data_ptr<scalar_t>(),
                    x.data_ptr<scalar_t>(),
                    mask.data_ptr<unsigned char>(),
                    dim_size,
                    batch_size
                );
            }));

            return output;
        }
        """

        self.custom_cumsum = load_inline(
            name="custom_cumsum",
            cpp_sources=custom_cumsum_source,
            functions=["custom_cumsum_cuda"],
            verbose=True,
        )

    def forward(self, x, mask):
        dim_size = x.size(self.dim)
        batch_size = x.size(0)
        return self.custom_cumsum.custom_cumsum_cuda(x, mask, dim_size, batch_size)