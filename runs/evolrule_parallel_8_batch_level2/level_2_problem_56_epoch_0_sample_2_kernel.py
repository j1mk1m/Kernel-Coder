import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

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
            int b = blockIdx.x;
            if (b >= batch_size) return;

            extern __shared__ float shared_mem[];
            int tid = threadIdx.x;
            int num_threads = blockDim.x;

            float sum = 0.0f;
            for (int j = tid; j < hidden_size; j += num_threads) {
                float val = input[b * hidden_size + j];
                sum += 1.0f / (1.0f + expf(-val));
            }

            shared_mem[tid] = sum;
            __syncthreads();

            for (int s = num_threads / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_mem[tid] += shared_mem[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[b] = shared_mem[0];
            }
        }

        torch::Tensor sigmoid_sum_cuda(torch::Tensor input) {
            auto batch_size = input.size(0);
            auto hidden_size = input.size(1);
            auto output = torch::empty({batch_size, 1}, input.options());

            int block_size = 256;
            int shared_size = block_size * sizeof(float);

            const dim3 grid(batch_size);
            const dim3 block(block_size);

            sigmoid_sum_kernel<<<grid, block, shared_size, torch::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                hidden_size
            );

            return output;
        }
        """

        sigmoid_sum_cpp = "torch::Tensor sigmoid_sum_cuda(torch::Tensor input);"

        self.sigmoid_sum = load_inline(
            name="sigmoid_sum",
            cpp_sources=sigmoid_sum_cpp,
            cuda_sources=sigmoid_sum_source,
            functions=["sigmoid_sum_cuda"],
            verbose=True
        )

    def forward(self, x):
        linear_out = self.linear(x)
        return self.sigmoid_sum.sigmoid_sum_cuda(linear_out)

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size]