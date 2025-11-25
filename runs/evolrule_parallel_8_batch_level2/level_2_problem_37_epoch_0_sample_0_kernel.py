import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        # Initialize weights and bias similarly to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        # Initialize weight with the same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Load fused CUDA kernel for matmul, swish, bias addition
        fused_kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <math.h>

        template <typename scalar_t>
        __device__ scalar_t swish(scalar_t x) {
            return x / (1.0 + exp(-x));
        }

        template <typename scalar_t>
        __global__ void fused_forward_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            scalar_t* __restrict__ output,
            int batch_size,
            int in_features,
            int out_features
        ) {
            int batch_idx = blockIdx.x;
            int out_idx = threadIdx.x;

            if (out_idx >= out_features) return;

            scalar_t sum = 0.0;
            for (int in_idx = 0; in_idx < in_features; ++in_idx) {
                sum += input[batch_idx * in_features + in_idx] * 
                    weight[out_idx * in_features + in_idx];
            }

            sum = swish(sum + bias[out_idx]);
            output[batch_idx * out_features + out_idx] = sum;
        }

        torch::Tensor fused_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias
        ) {
            const int batch_size = input.size(0);
            const int in_features = input.size(1);
            const int out_features = weight.size(0);
            
            auto output = torch::empty({batch_size, out_features}, 
                torch::dtype(input.dtype()).device(input.device()));

            const int threads = 256;
            const int blocks = batch_size;

            AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_forward_cuda", ([&] {
                fused_forward_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    in_features,
                    out_features);
            }));

            return output;
        }
        """
        
        self.fused_forward = load_inline(
            name="fused_forward",
            cpp_sources="",
            cuda_sources=fused_kernel_source,
            functions=["fused_forward"],
            verbose=True
        )

    def forward(self, x):
        # Fused matmul + Swish + bias addition
        x = self.fused_forward.fused_forward(x, self.weight, self.bias)
        
        # Reshape for GroupNorm (since GroupNorm expects channels first)
        x = x.view(x.size(0), -1)  # Ensure contiguous
        x = self.group_norm(x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
        return x

def get_inputs():
    batch_size = 32768
    in_features = 1024
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    in_features = 1024
    out_features = 4096
    num_groups = 64
    bias_shape = (4096,)
    return [in_features, out_features, num_groups, bias_shape]