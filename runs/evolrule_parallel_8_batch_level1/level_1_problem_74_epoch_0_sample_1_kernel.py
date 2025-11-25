import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Define and compile the custom CUDA kernel
        self.conv_transpose_1d_kernel = load_inline(
            name="conv_transpose_1d_kernel",
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __global__ void conv_transpose_1d_kernel(
                const scalar_t* __restrict__ input,
                const scalar_t* __restrict__ weight,
                scalar_t* __restrict__ output,
                int batch_size,
                int in_channels,
                int out_channels,
                int kernel_size,
                int input_length,
                int output_length,
                int stride,
                int padding,
                int dilation) {{
                int batch_idx = blockIdx.x;
                int out_channel = blockIdx.y;
                int in_channel = threadIdx.x;
                
                if (in_channel >= in_channels) return;
                
                int output_offset = batch_idx * out_channels * output_length + out_channel * output_length;
                const int weight_offset = out_channel * in_channels * kernel_size + in_channel * kernel_size;
                
                for (int pos = 0; pos < output_length; pos += blockDim.y) {{
                    int current_pos = pos + threadIdx.y;
                    if (current_pos >= output_length) continue;
                    
                    scalar_t sum = 0;
                    for (int k = 0; k < kernel_size; ++k) {{
                        int dilated_k = k * dilation;
                        int input_pos = (current_pos - dilated_k - padding) / stride;
                        if (input_pos < 0 || input_pos >= input_length) continue;
                        if ((current_pos - dilated_k - padding) % stride != 0) continue;
                        
                        int weight_idx = weight_offset + k;
                        int input_idx = batch_idx * in_channels * input_length + in_channel * input_length + input_pos;
                        sum += weight[weight_idx] * input[input_idx];
                    }}
                    atomicAdd(&output[output_offset + current_pos], sum);
                }}
            }}
            
            at::Tensor conv_transpose_1d_cuda(
                at::Tensor input,
                at::Tensor weight,
                int stride,
                int padding,
                int dilation) {{
                auto batch_size = input.size(0);
                auto in_channels = input.size(1);
                auto output_length = (input.size(2) - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
                auto output = at::empty({{batch_size, out_channels, output_length}}, input.options());
                
                dim3 threads(32, 32); // Tuned thread configuration
                dim3 blocks(batch_size, out_channels);
                
                AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_1d_cuda", ([&] {{
                    conv_transpose_1d_kernel<scalar_t><<<blocks, threads>>>(
                        input.data<scalar_t>(),
                        weight.data<scalar_t>(),
                        output.data<scalar_t>(),
                        batch_size,
                        in_channels,
                        out_channels,
                        kernel_size,
                        input.size(2),
                        output.size(2),
                        stride,
                        padding,
                        dilation);
                }}));
                
                return output;
            }}
            """,
            functions=["conv_transpose_1d_cuda"],
            verbose=False
        )
        
    def forward(self, x):
        output = self.conv_transpose_1d_kernel.conv_transpose_1d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output