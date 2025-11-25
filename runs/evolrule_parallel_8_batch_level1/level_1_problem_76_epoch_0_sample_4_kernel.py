import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and optional bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias_term = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias_term = None
            
        # Initialize the CUDA kernel
        self.conv1d_cuda = load_inline(
            name="conv1d_cuda",
            cpp_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            #include <vector>
            
            at::Tensor conv1d_forward(
                const at::Tensor& input,
                const at::Tensor& weight,
                const c10::optional<at::Tensor>& bias,
                int stride,
                int dilation
            );
            """,
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            #include <vector>
            
            __global__ void conv1d_kernel(
                const float* __restrict__ input,
                const float* __restrict__ weight,
                float* __restrict__ output,
                const int batch_size,
                const int in_channels,
                const int out_channels,
                const int kernel_size,
                const int input_length,
                const int output_length,
                const int stride,
                const int dilation,
                const bool has_bias,
                const float* __restrict__ bias
            ) {{
                // Calculate output position
                int b = blockIdx.x;
                int oc = blockIdx.y;
                int pos_out = threadIdx.x;
                
                if (pos_out >= output_length)
                    return;
                
                // Compute the output value for this (b, oc, pos_out)
                float acc = 0.0;
                for (int ic = 0; ic < in_channels; ic++) {{
                    for (int k = 0; k < kernel_size; k++) {{
                        // Compute input position
                        int pos_in = pos_out * stride + k * dilation;
                        if (pos_in < input_length) {{
                            acc += weight[oc * in_channels * kernel_size + ic * kernel_size + k] *
                                   input[b * in_channels * input_length + ic * input_length + pos_in];
                        }}
                    }}
                }}
                
                if (has_bias) {{
                    acc += bias[oc];
                }}
                
                output[b * out_channels * output_length + oc * output_length + pos_out] = acc;
            }}
            
            at::Tensor conv1d_forward(
                const at::Tensor& input,
                const at::Tensor& weight,
                const c10::optional<at::Tensor>& bias_opt,
                int stride,
                int dilation
            ) {{
                const int batch_size = input.size(0);
                const int in_channels = input.size(1);
                const int out_channels = weight.size(0);
                const int kernel_size = weight.size(2);
                const int input_length = input.size(2);
                
                // Compute output length
                const int output_length = 
                    (input_length - dilation * (kernel_size - 1) - 1) / stride + 1;
                
                at::Tensor output = at::empty({{batch_size, out_channels, output_length}}, input.options());
                
                const bool has_bias = bias_opt.has_value();
                const float* bias_data = has_bias ? bias_opt.value().data_ptr<float>() : nullptr;
                
                // Define grid and block dimensions
                dim3 blockDim(output_length); // Each thread handles one output position
                dim3 gridDim(batch_size, out_channels); // Each block handles a (batch, output_channel) pair
                
                // Launch kernel
                conv1d_kernel<<<gridDim, blockDim>>>(
                    input.data_ptr<float>(),
                    weight.data_ptr<float>(),
                    output.data_ptr<float>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    kernel_size,
                    input_length,
                    output_length,
                    stride,
                    dilation,
                    has_bias,
                    bias_data
                );
                
                cudaDeviceSynchronize();
                return output;
            }}
            """,
            functions=[
                "at::Tensor conv1d_forward(const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&, int, int)"
            ],
            verbose=False
        )
        
        # Initialize weights similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_term is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_term, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_term if self.bias else None
        return self.conv1d_cuda.conv1d_forward(
            x.cuda(),
            self.weight.cuda(),
            bias,
            self.stride,
            self.dilation
        )

def get_inputs():
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]