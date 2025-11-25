import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

        # Define fused element-wise CUDA kernel
        elementwise_cpp = """
        torch::Tensor fused_min_subtract_cuda(torch::Tensor input, float c);
        """
        elementwise_cuda = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void fused_min_subtract(const float* input, float c, float* output, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] = min(input[idx] - c, 0.0f);
            }
        }

        torch::Tensor fused_min_subtract_cuda(torch::Tensor input, float c) {
            auto size = input.numel();
            auto output = torch::zeros_like(input);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            fused_min_subtract<<<num_blocks, block_size>>>(input.data_ptr<float>(), c, output.data_ptr<float>(), size);

            return output;
        }
        """
        
        # Compile the fused kernel
        self.fused_min_subtract = load_inline(
            name="fused_min_subtract",
            cpp_sources=elementwise_cpp,
            cuda_sources=elementwise_cuda,
            functions=["fused_min_subtract_cuda"],
            verbose=True,
            extra_cflags=[""],
            extra_ldflags=[""]
        )

    def forward(self, x):
        x = self.linear(x)
        c_val = self.constant.item()
        x = self.fused_min_subtract.fused_min_subtract_cuda(x, c_val)
        return x

# Required functions for model creation
batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, constant]