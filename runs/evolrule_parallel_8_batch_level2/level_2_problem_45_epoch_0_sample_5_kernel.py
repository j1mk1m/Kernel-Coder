import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Sigmoid kernel
        sigmoid_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void sigmoid_kernel(const float* input, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = 1.0f / (1.0f + expf(-input[idx]));
            }
        }

        torch::Tensor sigmoid_cuda(torch::Tensor input) {
            auto size = input.numel();
            auto output = torch::empty_like(input);
            
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            sigmoid_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
            return output;
        }
        """
        
        # Corrected LogSumExp kernel with proper shared memory and wrapper
        logsumexp_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <limits>

        __global__ void logsumexp_kernel(const float* input, float* output, int batch_size, int features) {
            extern __shared__ float shared[];
            
            int sample = blockIdx.x;
            int tid = threadIdx.x;
            int block_size = blockDim.x;
            
            // Load data into shared memory
            for (int i = tid; i < features; i += block_size) {
                shared[i] = input[sample * features + i];
            }
            __syncthreads();
            
            // Compute max value
            float max_val = -std::numeric_limits<float>::infinity();
            for (int i = tid; i < features; i += block_size) {
                if (shared[i] > max_val) {
                    max_val = shared[i];
                }
            }
            
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    max_val = max(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, s));
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                output[sample] = max_val;
            }
            __syncthreads();
            
            // Compute sum(exp(x - max))
            float sum = 0.0f;
            for (int i = tid; i < features; i += block_size) {
                sum += expf(shared[i] - max_val);
            }
            
            for (int s = block_size / 2; s > 0; s >>= 1) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
            }
            
            if (tid == 0) {
                output[sample] = logf(sum) + max_val;
            }
        }

        torch::Tensor logsumexp_cuda(torch::Tensor input, int batch_size, int features) {
            auto output = torch::empty({batch_size}, input.options());
            const int block_size = 256; // Threads per block
            const size_t shared_mem = features * sizeof(float);
            
            logsumexp_kernel<<<batch_size, block_size, shared_mem>>>(
                input.data_ptr<float>(), 
                output.data_ptr<float>(), 
                batch_size, 
                features
            );
            return output;
        }
        """
        
        # Compile sigmoid kernel
        sigmoid_cpp = "torch::Tensor sigmoid_cuda(torch::Tensor input);"
        self.sigmoid = load_inline(
            name="sigmoid",
            cpp_sources=sigmoid_cpp,
            cuda_sources=sigmoid_source,
            functions=["sigmoid_cuda"],
            verbose=True
        )
        
        # Compile logsumexp kernel
        logsumexp_cpp = "torch::Tensor logsumexp_cuda(torch::Tensor input, int batch_size, int features);"
        self.logsumexp = load_inline(
            name="logsumexp",
            cpp_sources=logsumexp_cpp,
            cuda_sources=logsumexp_source,
            functions=["logsumexp_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid.sigmoid_cuda(x)
        x = self.linear2(x)
        batch_size, features = x.size()
        return self.logsumexp.logsumexp_cuda(x, batch_size, features)

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, output_size]