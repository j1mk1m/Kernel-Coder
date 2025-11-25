import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        # Initialize weights and biases with same method as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize using default nn.Linear initialization (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain('leaky_relu', negative_slope), mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.bias, 0)
        
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        # Inline CUDA kernel code for fused GEMM + scale + leaky_relu
        self.fused_kernel = load_inline(
            name="fused_gemm_scale_leakyrelu",
            cpp_sources=f"""
                torch::Tensor fused_gemm_scale_leakyrelu_cuda(
                    torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    const float multiplier,
                    const float negative_slope);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <cuda_fp16.h>
                #define THREADS 256

                template<typename T>
                __global__ void fused_kernel(
                    const T* __restrict__ input,
                    const T* __restrict__ weight,
                    const T* __restrict__ bias,
                    T* __restrict__ output,
                    int batch_size, int in_features, int out_features,
                    const T multiplier, const T negative_slope) {{
                    
                    int batch_id = blockIdx.x;
                    int out_id = threadIdx.x;
                    
                    T sum = bias[out_id];
                    
                    for (int i = 0; i < in_features; i += THREADS) {{
                        int idx = i + threadIdx.x;
                        T val = (idx < in_features) ? input[batch_id * in_features + idx] * weight[out_id * in_features + idx] : 0;
                        sum += val;
                    }}
                    
                    __shared__ T shared_sum[THREADS];
                    shared_sum[threadIdx.x] = sum;
                    __syncthreads();
                    
                    for (int offset = THREADS/2; offset > 0; offset >>= 1) {{
                        if (threadIdx.x < offset) {{
                            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
                        }}
                        __syncthreads();
                    }}
                    
                    if (threadIdx.x == 0) {{
                        sum = shared_sum[0] * multiplier;
                        output[batch_id * out_features + out_id] = sum > 0 ? sum : sum * negative_slope;
                    }}
                }}

                torch::Tensor fused_gemm_scale_leakyrelu_cuda(
                    torch::Tensor input,
                    torch::Tensor weight,
                    torch::Tensor bias,
                    const float multiplier,
                    const float negative_slope) {{
                    
                    const int batch_size = input.size(0);
                    const int in_features = input.size(1);
                    const int out_features = weight.size(0);
                    
                    auto output = torch::empty({{batch_size, out_features}}, 
                                              input.options());
                    
                    dim3 blocks(batch_size);
                    dim3 threads(min(THREADS, out_features));
                    
                    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_gemm_scale_leakyrelu_cuda", ([&] {
                        fused_kernel<scalar_t><<<blocks, threads>>>(
                            input.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(),
                            bias.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size, in_features, out_features,
                            multiplier, negative_slope);
                    }));
                    
                    cudaDeviceSynchronize();
                    return output;
                }}
            """,
            functions=["fused_gemm_scale_leakyrelu_cuda"],
            verbose=True
        )

    def forward(self, x):
        # Ensure inputs and parameters are on the same device
        x = x.to(self.weight.device)
        return self.fused_kernel.fused_gemm_scale_leakyrelu_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )

# Check that the inputs are correctly generated
def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]