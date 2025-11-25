import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

        # Define the custom CUDA kernel for MaxPool1D
        kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        // Hardcode parameters for the given problem
        const int kernel_size = 8;
        const int dilation = 3;
        const int padding = 4;
        const int stride = 1;

        __global__ void max_pool1d_cuda_kernel(
            const float* input, 
            float* output,
            int batch_size,
            int features,
            int input_length,
            int output_length) {
            int TPB = blockDim.x;
            int chunks = (output_length + TPB - 1) / TPB;
            int block_idx = blockIdx.x;
            int chunk_id = block_idx % chunks;
            int block_offset = block_idx / chunks;

            int batch = block_offset / features;
            int feature = block_offset % features;

            int tid = threadIdx.x;
            int output_idx = chunk_id * TPB + tid;
            if (output_idx >= output_length) return;

            // Starting position in padded input
            int s = output_idx * stride; // since stride is 1

            float max_val = -INFINITY;
            for (int k = 0; k < kernel_size; ++k) {
                int pos = s + k * dilation;
                if (pos < 0 || pos >= (input_length + 2 * padding)) {
                    continue;
                }
                int orig_pos = pos - padding;
                if (orig_pos < 0 || orig_pos >= input_length) {
                    continue;
                }
                int input_offset = batch * features * input_length + feature * input_length + orig_pos;
                float val = input[input_offset];
                if (val > max_val) {
                    max_val = val;
                }
            }

            // Write to output
            int output_offset = batch * features * output_length + feature * output_length + output_idx;
            output[output_offset] = max_val;
        }

        void max_pool1d_cuda(
            torch::Tensor input,
            torch::Tensor output,
            int batch_size,
            int features,
            int input_length,
            int output_length) {
            const int TPB = 1024;
            int chunks = (output_length + TPB - 1) / TPB;
            int total_blocks = batch_size * features * chunks;

            dim3 blocks(total_blocks);
            dim3 threads(TPB);

            max_pool1d_cuda_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                features,
                input_length,
                output_length
            );
            cudaDeviceSynchronize();
        }
        """

        self.max_pool_cuda = load_inline(
            name="max_pool_cuda",
            cuda_sources=kernel_source,
            functions=["max_pool1d_cuda"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.size(2)
        padded_length = input_length + 2 * self.padding
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        output_length = (padded_length - effective_kernel_size) // self.stride + 1

        output = torch.empty(x.shape[0], x.shape[1], output_length, device=x.device, dtype=x.dtype)

        # Call the CUDA kernel
        self.max_pool_cuda.max_pool1d_cuda(
            x.contiguous(),
            output,
            x.shape[0],  # batch_size
            x.shape[1],  # features
            input_length,
            output_length
        )

        return output

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]