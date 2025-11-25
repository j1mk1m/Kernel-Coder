import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        # Define and load the custom CUDA kernel
        sigmoid_sum_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void sigmoid_sum_kernel(
            const float* input,
            float* output,
            int batch_size,
            int hidden_size
        ) {
            extern __shared__ float sdata[];
            int b = blockIdx.x;
            if (b >= batch_size) return;

            int tid = threadIdx.x;
            int nthreads = blockDim.x;

            float sum = 0.0f;
            for (int h = tid; h < hidden_size; h += nthreads) {
                float val = input[b * hidden_size + h];
                val = 1.0f / (1.0f + expf(-val));
                sum += val;
            }

            sdata[tid] = sum;
            __syncthreads();

            for (int s = nthreads / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[b] = sdata[0];
            }
        }

        torch::Tensor sigmoid_sum_cuda(torch::Tensor input, int hidden_size) {
            auto batch_size = input.size(0);
            auto output = torch::empty({batch_size, 1}, input.options());

            int threads_per_block = 256;
            int blocks_per_grid = batch_size;

            size_t shared_mem_size = threads_per_block * sizeof(float);

            sigmoid_sum_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                hidden_size
            );

            return output;
        }
        """
        sigmoid_sum_cpp = "torch::Tensor sigmoid_sum_cuda(torch::Tensor input, int hidden_size);"

        # Load the kernel
        self.sigmoid_sum = load_inline(
            name="sigmoid_sum",
            cpp_sources=sigmoid_sum_cpp,
            cuda_sources=sigmoid_sum_source,
            functions=["sigmoid_sum_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""]
        )

    def forward(self, x):
        x = self.linear(x)
        hidden_size = self.linear.out_features
        output = self.sigmoid_sum.sigmoid_sum_cuda(x, hidden_size)
        return output.view(-1, 1)

# Ensure the batch_size, input_size, hidden_size are same as original
batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size]