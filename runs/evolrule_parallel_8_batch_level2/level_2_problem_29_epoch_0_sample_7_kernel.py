import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Custom CUDA kernel for applying Mish twice
        mish_twice_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        __global__ void mish_twice_kernel(const float* in, float* out, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float x = in[idx];
                // First Mish
                float abs_x = fabsf(x);
                float term1 = expf(-abs_x);
                term1 += 1.0f;
                term1 = logf(term1);
                float term2 = (x > 0) ? x : 0.0f;
                float z = term1 + term2;
                float tanhz = tanhf(z);
                x = x * tanhz;
                // Second Mish
                abs_x = fabsf(x);
                term1 = expf(-abs_x);
                term1 += 1.0f;
                term1 = logf(term1);
                term2 = (x > 0) ? x : 0.0f;
                z = term1 + term2;
                tanhz = tanhf(z);
                x = x * tanhz;
                out[idx] = x;
            }
        }

        torch::Tensor mish_twice_cuda(torch::Tensor in) {
            auto size = in.numel();
            auto out = torch::empty_like(in);

            const int block_size = 256;
            const int num_blocks = (size + block_size - 1) / block_size;

            mish_twice_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);

            return out;
        }
        """
        
        mish_twice_cpp_source = "torch::Tensor mish_twice_cuda(torch::Tensor in);"
        
        self.mish_twice = load_inline(
            name="mish_twice",
            cpp_sources=mish_twice_cpp_source,
            cuda_sources=mish_twice_source,
            functions=["mish_twice_cuda"],
            verbose=False,
            extra_cflags=["-O3"],
            extra_ldflags=[""]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.mish_twice.mish_twice_cuda(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]