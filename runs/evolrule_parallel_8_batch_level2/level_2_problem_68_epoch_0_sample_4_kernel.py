import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.constant = nn.Parameter(torch.tensor([constant], dtype=torch.float32))

        # Initialize weights and bias
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

        # Define and load custom CUDA kernels
        linear_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void linear_kernel(const scalar_t* input, const scalar_t* weight, const scalar_t* bias, scalar_t* output,
                                    int batch_size, int in_features, int out_features) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_features) return;

            int batch_idx = idx / out_features;
            int out_idx = idx % out_features;

            scalar_t sum = 0;
            for (int i = 0; i < in_features; i++) {
                sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
            }
            output[idx] = sum + bias[out_idx];
        }

        torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
            const int batch_size = input.size(0);
            const int in_features = input.size(1);
            const int out_features = weight.size(0);

            auto output = torch::empty({batch_size, out_features}, torch::device("cuda"), torch::dtype(torch::kFloat32));

            const int block_size = 256;
            const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

            linear_kernel<float><<<num_blocks, block_size>>>(
                input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                output.data_ptr<float>(), batch_size, in_features, out_features
            );

            return output;
        }
        """
        linear_cpp = "torch::Tensor linear_forward(torch::Tensor, torch::Tensor, torch::Tensor);"
        self.linear = load_inline(name="linear", cuda_sources=linear_source, cpp_sources=linear_cpp, functions=["linear_forward"], verbose=True)

        min_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void elementwise_min(const float* a, const float* b, float* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                out[idx] = min(a[idx], b[0]);
            }
        }

        torch::Tensor min_cuda(torch::Tensor a, torch::Tensor b) {
            auto size = a.numel();
            auto out = torch::empty_like(a);
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;
            elementwise_min<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
            return out;
        }
        """
        min_cpp = "torch::Tensor min_cuda(torch::Tensor, torch::Tensor);"
        self.min = load_inline(name="min", cuda_sources=min_source, cpp_sources=min_cpp, functions=["min_cuda"], verbose=True)

        subtract_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void elementwise_subtract(const float* a, const float* b, float* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                out[idx] = a[idx] - b[0];
            }
        }

        torch::Tensor subtract_cuda(torch::Tensor a, torch::Tensor b) {
            auto size = a.numel();
            auto out = torch::empty_like(a);
            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;
            elementwise_subtract<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
            return out;
        }
        """
        subtract_cpp = "torch::Tensor subtract_cuda(torch::Tensor, torch::Tensor);"
        self.sub = load_inline(name="subtract", cuda_sources=subtract_source, cpp_sources=subtract_cpp, functions=["subtract_cuda"], verbose=True)

    def forward(self, x):
        x = self.linear.linear_forward(x, self.weight, self.bias)
        x = self.min.min_cuda(x, self.constant)
        x = self.sub.subtract_cuda(x, self.constant)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, constant]