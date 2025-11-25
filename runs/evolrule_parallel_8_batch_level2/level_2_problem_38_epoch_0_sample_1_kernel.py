import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

        # Define custom CUDA kernel for combining clamp and softmax operations
        clamp_softmax_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        template <typename scalar_t>
        __global__ void clamp_softmax_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int batch_size, int channels, int spatial_size, float min_val, float max_val) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * channels * spatial_size) return;

            int b = idx / (channels * spatial_size);
            int c = (idx / spatial_size) % channels;
            int s = idx % spatial_size;

            scalar_t val = input[idx];
            val = max(val, min_val);
            val = min(val, max_val);

            // Compute max for numerical stability
            extern __shared__ scalar_t shared[];
            scalar_t* max_buf = shared;
            scalar_t* exp_buf = shared + spatial_size;

            // Each thread computes max for its spatial dimension
            if (threadIdx.x < spatial_size) {
                max_buf[threadIdx.x] = -INFINITY;
            }
            __syncthreads();

            for (int i = 0; i < spatial_size; i += blockDim.x) {
                int pos = c * spatial_size + i + threadIdx.x;
                if (pos < spatial_size) {
                    max_buf[threadIdx.x] = max(max_buf[threadIdx.x], input[b * channels * spatial_size + c * spatial_size + i + threadIdx.x]);
                }
            }
            __syncthreads();

            // Reduce to find the maximum
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    max_buf[threadIdx.x] = max(max_buf[threadIdx.x], max_buf[threadIdx.x + s]);
                }
                __syncthreads();
            }
            scalar_t max_val = max_buf[0];
            __syncthreads();

            // Compute exponentials
            exp_buf[threadIdx.x] = exp(val - max_val);
            __syncthreads();

            // Sum reduction for denominator
            scalar_t sum = 0;
            for (int i = 0; i < spatial_size; i++) {
                sum += exp_buf[i];
            }
            __syncthreads();

            // Compute softmax and clamp
            output[idx] = exp_buf[threadIdx.x] / sum;
        }

        torch::Tensor clamp_softmax_cuda(torch::Tensor input, float min_val, float max_val) {
            auto batch_size = input.size(0);
            auto channels = input.size(1);
            auto spatial_size = input.size(2) * input.size(3) * input.size(4);

            auto output = torch::zeros_like(input);
            int block_size = 256;
            int num_blocks = (batch_size * channels * spatial_size + block_size - 1) / block_size;

            // The shared memory size is 2 * spatial_size
            auto shared_mem = 2 * spatial_size * sizeof(float);

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "clamp_softmax_cuda", ([&] {
                clamp_softmax_kernel<scalar_t><<<num_blocks, block_size, shared_mem>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    channels,
                    spatial_size,
                    min_val,
                    max_val);
            }));

            return output;
        }
        """

        clamp_softmax_cpp = "torch::Tensor clamp_softmax_cuda(torch::Tensor input, float min_val, float max_val);"
        self.clamp_softmax = load_inline(
            name="clamp_softmax",
            cpp_sources=clamp_softmax_cpp,
            cuda_sources=clamp_softmax_source,
            functions=["clamp_softmax_cuda"],
            verbose=True,
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        
        # Combine clamp and softmax into a single kernel
        x = self.clamp_softmax.clamp_softmax_cuda(x, self.clamp_min, self.clamp_max)
        x = x * self.scale
        return x

# Ensure the original get_inputs and get_init_inputs remain unchanged as per the problem statement
def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]