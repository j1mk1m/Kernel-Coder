import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

        # Define fused CUDA kernels
        self.fused_gemm_div_sum_scale = load_inline(
            name="fused_gemm_div_sum_scale",
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            #include <cuda_fp16.h>
            #include <mma.h>

            template<typename T>
            __global__ void fused_gemm_div_sum_scale_kernel(
                const T* __restrict__ input,
                const T* __restrict__ weight,
                T* output,
                int batch_size,
                int input_size,
                int hidden_size,
                T scaling_factor,
                T divisor) {{
                extern __shared__ unsigned char smem[];
                T* sdata = reinterpret_cast<T*>(smem);

                int batch = blockIdx.x;
                int tid = threadIdx.x;

                T sum = 0.0;
                #pragma unroll
                for (int hid = 0; hid < hidden_size; hid += blockDim.x) {{
                    int hid_idx = hid + tid;
                    if (hid_idx < hidden_size) {{
                        T val = 0.0;
                        #pragma unroll
                        for (int in = 0; in < input_size; in++) {{
                            val += input[batch * input_size + in] * weight[hid_idx * input_size + in];
                        }}
                        val /= divisor;
                        sum += val;
                    }}
                }}
                __syncthreads();

                // Parallel reduction for sum
                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
                    if (tid < stride) {{
                        sum += sdata[tid + stride];
                    }}
                    __syncthreads();
                }}

                if (tid == 0) {{
                    output[batch] = sum * scaling_factor;
                }}
            }}

            at::Tensor forward(
                at::Tensor input,
                at::Tensor weight,
                float scaling_factor,
                float divisor) {{
                const int batch_size = input.size(0);
                const int input_size = input.size(1);
                const int hidden_size = weight.size(0);
                auto output = at::empty({{batch_size, 1}}, input.options());

                int block_size = 256;
                int grid_size = batch_size;

                dim3 threads(block_size);
                dim3 blocks(grid_size);

                auto stream = at::cuda::getCurrentCUDAStream();
                fused_gemm_div_sum_scale_kernel<float>
                    <<<blocks, threads, 0, stream>>>(
                        input.data_ptr<float>(),
                        weight.data_ptr<float>(),
                        output.data_ptr<float>(),
                        batch_size,
                        input_size,
                        hidden_size,
                        scaling_factor,
                        divisor
                    );

                AT_CUDA_CHECK(cudaGetLastError());
                return output;
            }}
            """,
            functions=["forward"],
            verbose=True
        )

    def forward(self, x):
        # x: [batch_size, input_size]
        # weight: [hidden_size, input_size]
        # The fused kernel handles matmul, div, sum, and scaling
        return self.fused_gemm_div_sum_scale.forward(
            x,
            self.weight.t(),
            self.scaling_factor,
            2.0
        ).view(-1, 1)

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5