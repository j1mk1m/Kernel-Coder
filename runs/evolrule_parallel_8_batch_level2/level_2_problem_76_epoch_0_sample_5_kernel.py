import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class FusedGEMMReluBias(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(FusedGEMMReluBias, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.gemm_relu_bias_kernel = load_inline(
            name="fused_gemm_relu_bias",
            cuda Sources="""
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __global__ void fused_gemm_relu_bias_kernel(const scalar_t* __restrict__ input,
                                                        const scalar_t* __restrict__ weight,
                                                        const scalar_t* __restrict__ bias,
                                                        scalar_t* __restrict__ output,
                                                        int batch_size,
                                                        int in_features,
                                                        int out_features) {
                int batch_idx = blockIdx.x;
                int out_idx = threadIdx.x;

                if (out_idx >= out_features) return;

                scalar_t sum = 0;
                for (int i = 0; i < in_features; ++i) {
                    sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
                }
                sum += bias[out_idx];
                output[batch_idx * out_features + out_idx] = fmaxf(sum, 0);
            }

            torch::Tensor fused_gemm_relu_bias(torch::Tensor input,
                                               torch::Tensor weight,
                                               torch::Tensor bias) {
                const int threads = 256;
                const int blocks = input.size(0);

                auto output = torch::empty({input.size(0), weight.size(0)}, 
                                           dtype: input.dtype(), 
                                           device: input.device());

                AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_gemm_relu_bias", ([&] {
                    fused_gemm_relu_bias_kernel<scalar_t><<<blocks, threads>>>(
                        input.data<scalar_t>(),
                        weight.data<scalar_t>(),
                        bias.data<scalar_t>(),
                        output.data<scalar_t>(),
                        input.size(0),
                        input.size(1),
                        weight.size(0));
                }));

                return output;
            }
            """,
            cpp_sources="",
            functions=["fused_gemm_relu_bias"],
            verbose=True,
        )

    def forward(self, x):
        return self.gemm_relu_bias_kernel.fused_gemm_relu_bias(x, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.fused_layer = FusedGEMMReluBias(in_features, out_features, bias_shape)

    def forward(self, x):
        return self.fused_layer(x)

def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features, device="cuda")]

def get_init_inputs():
    return [1024, 8192, (8192,)]